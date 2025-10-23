"""
Importance Weighted Supervised Fine-Tuning (iw-SFT) Implementation
基于重要性加权的监督微调，用于语言模型的自适应训练

参考论文: Qin & Springenberg (2025) "Supervised Fine Tuning on Curated Data is Reinforcement Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class ImportanceWeightedSFTLoss(nn.Module):
    """
    重要性加权的SFT损失函数
    
    通过比较当前策略和参考策略的对数概率来计算重要性权重，
    从而实现自适应的数据加权
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        label_smoothing: float = 0.0,
        temperature: float = 1.0,
        clip_weight: Optional[Tuple[float, float]] = None
    ):
        """
        初始化iw-SFT损失
        
        Args:
            epsilon: 平滑因子，防止权重过大
            label_smoothing: 标签平滑系数
            temperature: 温度参数，控制权重的尖锐程度
            clip_weight: 权重裁剪范围 (min, max)
        """
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.clip_weight = clip_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        计算重要性加权的损失
        
        Args:
            logits: 模型输出的logits, shape: (batch_size, seq_len, vocab_size)
            labels: 目标标签, shape: (batch_size, seq_len)
            ref_log_probs: 参考模型的对数概率, shape: (batch_size, seq_len)
            return_weights: 是否返回重要性权重
            
        Returns:
            加权后的损失值，如果return_weights=True则返回(loss, weights)
        """
        # 计算标准交叉熵损失（每个token）
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取目标token的对数概率
        batch_size, seq_len, vocab_size = logits.shape
        labels_flat = labels.view(-1)
        log_probs_flat = log_probs.view(-1, vocab_size)
        
        # 计算每个token的负对数似然
        nll_loss = F.nll_loss(
            log_probs_flat,
            labels_flat,
            reduction='none',
            ignore_index=-100
        ).view(batch_size, seq_len)
        
        # 计算重要性权重
        if ref_log_probs is not None:
            # iw = exp((ref_log_prob - current_log_prob) / temperature)
            # 等价于: iw = exp(ref_log_prob / T) / exp(current_log_prob / T)
            current_log_probs = -nll_loss
            
            # 计算重要性权重
            importance_weights = torch.exp(
                (ref_log_probs - current_log_probs) / self.temperature
            )
            
            # 裁剪权重
            if self.clip_weight is not None:
                importance_weights = torch.clamp(
                    importance_weights,
                    min=self.clip_weight[0],
                    max=self.clip_weight[1]
                )
        else:
            # 如果没有参考模型，使用均匀权重
            importance_weights = torch.ones_like(nll_loss)
        
        # 创建padding mask
        padding_mask = (labels != -100).float()
        
        # 应用重要性权重
        weighted_loss = nll_loss * importance_weights * padding_mask
        
        # 计算平均损失
        num_tokens = padding_mask.sum()
        loss = weighted_loss.sum() / (num_tokens + 1e-8)
        
        if return_weights:
            return loss, importance_weights
        return loss


class QualitySampledDataLoader:
    """
    基于质量分数的数据采样器
    
    根据数据的质量分数（如reward）调整采样概率
    """
    
    def __init__(
        self,
        dataset,
        quality_scores: np.ndarray,
        batch_size: int,
        temperature: float = 1.0,
        min_prob: float = 0.01
    ):
        """
        初始化质量采样数据加载器
        
        Args:
            dataset: 数据集
            quality_scores: 每个样本的质量分数
            batch_size: 批次大小
            temperature: 温度参数，控制采样分布的尖锐程度
            min_prob: 最小采样概率，防止某些样本永远不被采样
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.min_prob = min_prob
        
        # 计算采样概率
        self.sampling_probs = self._compute_sampling_probs(quality_scores)
    
    def _compute_sampling_probs(self, quality_scores: np.ndarray) -> np.ndarray:
        """
        根据质量分数计算采样概率
        
        Args:
            quality_scores: 质量分数数组
            
        Returns:
            采样概率数组
        """
        # 归一化质量分数到[0, 1]
        scores = quality_scores - quality_scores.min()
        if scores.max() > 0:
            scores = scores / scores.max()
        
        # 应用温度缩放
        scores = scores / self.temperature
        
        # 转换为概率（softmax）
        exp_scores = np.exp(scores - scores.max())  # 数值稳定
        probs = exp_scores / exp_scores.sum()
        
        # 应用最小概率
        probs = np.maximum(probs, self.min_prob)
        probs = probs / probs.sum()  # 重新归一化
        
        return probs
    
    def sample_batch(self) -> Tuple[list, np.ndarray]:
        """
        采样一个批次
        
        Returns:
            (样本列表, 重要性权重)
        """
        # 根据概率采样索引
        indices = np.random.choice(
            len(self.dataset),
            size=self.batch_size,
            replace=False,
            p=self.sampling_probs
        )
        
        # 获取样本
        batch = [self.dataset[i] for i in indices]
        
        # 计算重要性采样权重
        sampled_probs = self.sampling_probs[indices]
        importance_weights = 1.0 / (len(self.dataset) * sampled_probs)
        importance_weights = importance_weights / importance_weights.max()
        
        return batch, importance_weights


class AdaptiveDataWeighter:
    """
    自适应数据加权器
    
    根据模型在不同数据上的表现动态调整数据权重
    类似于课程学习和难例挖掘的结合
    """
    
    def __init__(
        self,
        num_samples: int,
        initial_weight: float = 1.0,
        momentum: float = 0.9,
        update_frequency: int = 100
    ):
        """
        初始化自适应数据加权器
        
        Args:
            num_samples: 样本总数
            initial_weight: 初始权重
            momentum: 动量系数，用于平滑权重更新
            update_frequency: 权重更新频率（每N个样本）
        """
        self.num_samples = num_samples
        self.weights = np.ones(num_samples) * initial_weight
        self.momentum = momentum
        self.update_frequency = update_frequency
        
        # 累积统计
        self.loss_history = [[] for _ in range(num_samples)]
        self.update_count = 0
    
    def get_weights(self, indices: np.ndarray) -> np.ndarray:
        """
        获取指定索引的权重
        
        Args:
            indices: 样本索引
            
        Returns:
            权重数组
        """
        return self.weights[indices]
    
    def update(self, indices: np.ndarray, losses: np.ndarray):
        """
        根据损失更新权重
        
        Args:
            indices: 样本索引
            losses: 对应的损失值
        """
        # 记录损失历史
        for idx, loss in zip(indices, losses):
            self.loss_history[idx].append(loss)
        
        self.update_count += len(indices)
        
        # 定期更新权重
        if self.update_count >= self.update_frequency:
            self._update_weights()
            self.update_count = 0
    
    def _update_weights(self):
        """更新所有样本的权重"""
        new_weights = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            if len(self.loss_history[i]) > 0:
                # 使用最近的平均损失作为难度指标
                avg_loss = np.mean(self.loss_history[i][-10:])
                new_weights[i] = avg_loss
            else:
                new_weights[i] = self.weights[i]
        
        # 归一化权重
        if new_weights.max() > new_weights.min():
            new_weights = (new_weights - new_weights.min()) / (new_weights.max() - new_weights.min())
            new_weights = new_weights + 0.1  # 防止权重为0
        else:
            new_weights = np.ones_like(new_weights)
        
        # 使用动量更新
        self.weights = self.momentum * self.weights + (1 - self.momentum) * new_weights
        
        # 归一化
        self.weights = self.weights / self.weights.mean()


class RewardWeightedSampler:
    """
    基于Reward的加权采样器
    
    根据rollout的reward分布来决定采样概率
    类似于RWR (Reward Weighted Regression)
    """
    
    def __init__(
        self,
        rewards: np.ndarray,
        temperature: float = 1.0,
        use_rank: bool = False
    ):
        """
        初始化reward加权采样器
        
        Args:
            rewards: 每个样本的reward值
            temperature: 温度参数
            use_rank: 是否使用排名而非原始reward
        """
        self.rewards = rewards
        self.temperature = temperature
        self.use_rank = use_rank
        
        # 计算采样权重
        self.weights = self._compute_weights()
    
    def _compute_weights(self) -> np.ndarray:
        """计算采样权重"""
        if self.use_rank:
            # 使用排名
            ranks = np.argsort(np.argsort(self.rewards))
            scores = ranks.astype(float)
        else:
            # 使用原始reward
            scores = self.rewards.copy()
        
        # 归一化到[0, 1]
        scores = scores - scores.min()
        if scores.max() > 0:
            scores = scores / scores.max()
        
        # 应用温度
        scores = scores / self.temperature
        
        # 转换为权重（指数）
        weights = np.exp(scores - scores.max())
        weights = weights / weights.sum()
        
        return weights
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样一个批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (采样索引, 重要性权重)
        """
        indices = np.random.choice(
            len(self.rewards),
            size=batch_size,
            replace=False,
            p=self.weights
        )
        
        # 计算重要性采样权重
        sampled_weights = self.weights[indices]
        importance_weights = 1.0 / (len(self.rewards) * sampled_weights)
        importance_weights = importance_weights / importance_weights.max()
        
        return indices, importance_weights


if __name__ == "__main__":
    print("=== 测试ImportanceWeightedSFTLoss ===")
    
    # 创建模拟数据
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    ref_log_probs = torch.randn(batch_size, seq_len) * 0.1
    
    # 测试损失函数
    loss_fn = ImportanceWeightedSFTLoss(epsilon=0.1)
    loss, weights = loss_fn(logits, labels, ref_log_probs, return_weights=True)
    
    print(f"损失值: {loss.item():.4f}")
    print(f"重要性权重形状: {weights.shape}")
    print(f"权重统计: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    
    print("\n=== 测试RewardWeightedSampler ===")
    
    # 创建模拟reward
    rewards = np.random.randn(100)
    sampler = RewardWeightedSampler(rewards, temperature=0.5)
    
    indices, importance_weights = sampler.sample(batch_size=10)
    print(f"采样索引: {indices}")
    print(f"重要性权重: {importance_weights}")
    
    print("\n=== 测试AdaptiveDataWeighter ===")
    
    weighter = AdaptiveDataWeighter(num_samples=100, update_frequency=50)
    
    # 模拟训练过程
    for epoch in range(3):
        indices = np.random.choice(100, size=20, replace=False)
        losses = np.random.rand(20)
        
        weights = weighter.get_weights(indices)
        weighter.update(indices, losses)
        
        print(f"Epoch {epoch}: 权重统计 - min={weights.min():.4f}, max={weights.max():.4f}")

