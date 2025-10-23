"""
在线强化学习中的自适应采样方法
Online RL Adaptive Sampling Methods

解决在线策略（on-policy）RL中使用重要性采样的挑战：
1. 分布偏移问题
2. PPO/GRPO的重要性采样修正
3. 自适应采样与策略更新的协调

参考论文:
- Corrado et al. (2023) "On-Policy Policy Gradient RL Without On-Policy Sampling"
- DeepSeek-R1, OpenAI o1 等使用的GRPO方法
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from collections import deque


class OnPolicyAdaptiveSampler:
    """
    在线策略自适应采样器
    
    核心思想：
    1. 维护一个滑动窗口的经验池
    2. 根据advantage值动态调整采样权重
    3. 使用重要性采样权重修正分布偏移
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        clip_ratio: float = 0.2,
        advantage_clip: float = 10.0,
        use_adaptive_weight: bool = True,
        temperature: float = 1.0
    ):
        """
        初始化在线策略自适应采样器
        
        Args:
            buffer_size: 经验池大小
            clip_ratio: PPO的裁剪比率
            advantage_clip: advantage裁剪范围
            use_adaptive_weight: 是否使用自适应权重
            temperature: 温度参数，控制采样分布的尖锐程度
        """
        self.buffer_size = buffer_size
        self.clip_ratio = clip_ratio
        self.advantage_clip = advantage_clip
        self.use_adaptive_weight = use_adaptive_weight
        self.temperature = temperature
        
        # 经验池
        self.experiences = deque(maxlen=buffer_size)
        self.sampling_weights = deque(maxlen=buffer_size)
    
    def add_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        advantages: np.ndarray
    ):
        """
        添加一条轨迹到经验池
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            log_probs: 对数概率序列
            values: 价值估计序列
            advantages: advantage序列
        """
        trajectory_length = len(states)
        
        for i in range(trajectory_length):
            experience = {
                'state': states[i],
                'action': actions[i],
                'reward': rewards[i],
                'log_prob': log_probs[i],
                'value': values[i],
                'advantage': advantages[i]
            }
            
            # 计算初始采样权重（基于advantage的绝对值）
            if self.use_adaptive_weight:
                weight = np.abs(advantages[i]) + 1e-6
            else:
                weight = 1.0
            
            self.experiences.append(experience)
            self.sampling_weights.append(weight)
    
    def sample_batch(
        self,
        batch_size: int,
        current_policy_log_probs: Optional[np.ndarray] = None
    ) -> Tuple[Dict, np.ndarray]:
        """
        采样一个批次
        
        Args:
            batch_size: 批次大小
            current_policy_log_probs: 当前策略的对数概率（用于重要性采样）
            
        Returns:
            (批次数据, 重要性采样权重)
        """
        if len(self.experiences) < batch_size:
            raise ValueError(f"Buffer contains only {len(self.experiences)} samples, need {batch_size}")
        
        # 归一化采样权重
        weights = np.array(list(self.sampling_weights))
        weights = weights / self.temperature
        probs = weights / weights.sum()
        
        # 采样索引
        indices = np.random.choice(
            len(self.experiences),
            size=batch_size,
            replace=False,
            p=probs
        )
        
        # 收集批次数据
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'advantages': []
        }
        
        for idx in indices:
            exp = self.experiences[idx]
            batch['states'].append(exp['state'])
            batch['actions'].append(exp['action'])
            batch['rewards'].append(exp['reward'])
            batch['log_probs'].append(exp['log_prob'])
            batch['values'].append(exp['value'])
            batch['advantages'].append(exp['advantage'])
        
        # 转换为numpy数组
        for key in batch:
            batch[key] = np.array(batch[key])
        
        # 计算重要性采样权重
        sampled_probs = probs[indices]
        importance_weights = 1.0 / (len(self.experiences) * sampled_probs)
        
        # 归一化权重
        importance_weights = importance_weights / importance_weights.max()
        
        return batch, importance_weights


class GRPOAdaptiveSampler:
    """
    Group Relative Policy Optimization (GRPO) 自适应采样器
    
    GRPO是一种简化的在线RL算法，被DeepSeek-R1等模型使用
    核心思想：在一组样本内进行相对比较，而非绝对reward
    """
    
    def __init__(
        self,
        group_size: int = 8,
        advantage_temperature: float = 1.0,
        clip_range: float = 0.2,
        use_baseline: bool = True
    ):
        """
        初始化GRPO采样器
        
        Args:
            group_size: 每组的样本数量
            advantage_temperature: advantage计算的温度参数
            clip_range: 裁剪范围
            use_baseline: 是否使用组内baseline
        """
        self.group_size = group_size
        self.advantage_temperature = advantage_temperature
        self.clip_range = clip_range
        self.use_baseline = use_baseline
        
        # 存储当前组的数据
        self.current_group = []
    
    def add_sample(
        self,
        prompt: str,
        response: str,
        reward: float,
        log_prob: float
    ):
        """
        添加一个样本到当前组
        
        Args:
            prompt: 输入提示
            response: 模型响应
            reward: 奖励值
            log_prob: 对数概率
        """
        self.current_group.append({
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'log_prob': log_prob
        })
    
    def compute_group_advantages(self) -> np.ndarray:
        """
        计算组内的相对advantage
        
        Returns:
            advantage数组
        """
        if len(self.current_group) < self.group_size:
            raise ValueError(f"Group has only {len(self.current_group)} samples, need {self.group_size}")
        
        rewards = np.array([s['reward'] for s in self.current_group])
        
        if self.use_baseline:
            # 使用组内均值作为baseline
            baseline = rewards.mean()
            advantages = rewards - baseline
        else:
            # 直接使用reward
            advantages = rewards
        
        # 应用温度缩放
        advantages = advantages / self.advantage_temperature
        
        return advantages
    
    def sample_with_grpo_weights(self) -> Tuple[List[Dict], np.ndarray]:
        """
        根据GRPO策略计算采样权重
        
        Returns:
            (样本列表, 采样权重)
        """
        advantages = self.compute_group_advantages()
        
        # 将advantage转换为采样权重（使用softmax）
        exp_advantages = np.exp(advantages - advantages.max())  # 数值稳定
        weights = exp_advantages / exp_advantages.sum()
        
        return self.current_group, weights
    
    def clear_group(self):
        """清空当前组"""
        self.current_group = []


class PPOAdaptiveClipping:
    """
    PPO自适应裁剪机制
    
    动态调整PPO的裁剪范围，以适应不同阶段的训练需求
    """
    
    def __init__(
        self,
        initial_clip_range: float = 0.2,
        min_clip_range: float = 0.05,
        max_clip_range: float = 0.5,
        adaptation_rate: float = 0.01
    ):
        """
        初始化自适应裁剪
        
        Args:
            initial_clip_range: 初始裁剪范围
            min_clip_range: 最小裁剪范围
            max_clip_range: 最大裁剪范围
            adaptation_rate: 自适应调整速率
        """
        self.clip_range = initial_clip_range
        self.min_clip_range = min_clip_range
        self.max_clip_range = max_clip_range
        self.adaptation_rate = adaptation_rate
        
        # 统计信息
        self.kl_history = deque(maxlen=100)
        self.clip_fraction_history = deque(maxlen=100)
    
    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算PPO损失（带重要性采样修正）
        
        Args:
            old_log_probs: 旧策略的对数概率
            new_log_probs: 新策略的对数概率
            advantages: advantage值
            importance_weights: 重要性采样权重
            
        Returns:
            (损失值, 统计信息)
        """
        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 应用重要性采样权重
        if importance_weights is not None:
            ratio = ratio * importance_weights
        
        # PPO裁剪
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_range,
            1 + self.clip_range
        )
        
        # 计算损失
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        loss = -torch.min(loss1, loss2).mean()
        
        # 统计信息
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
            
            self.kl_history.append(kl_div)
            self.clip_fraction_history.append(clip_fraction)
        
        stats = {
            'loss': loss.item(),
            'kl_div': kl_div,
            'clip_fraction': clip_fraction,
            'clip_range': self.clip_range
        }
        
        return loss, stats
    
    def adapt_clip_range(self):
        """根据KL散度和裁剪比例自适应调整裁剪范围"""
        if len(self.kl_history) < 10:
            return
        
        avg_kl = np.mean(list(self.kl_history)[-10:])
        avg_clip_fraction = np.mean(list(self.clip_fraction_history)[-10:])
        
        # 如果KL散度过大，减小裁剪范围
        if avg_kl > 0.02:
            self.clip_range = max(
                self.min_clip_range,
                self.clip_range - self.adaptation_rate
            )
        # 如果裁剪比例过高，增大裁剪范围
        elif avg_clip_fraction > 0.5:
            self.clip_range = min(
                self.max_clip_range,
                self.clip_range + self.adaptation_rate
            )


class DistributionShiftDetector:
    """
    分布偏移检测器
    
    检测在线数据与离线数据之间的分布偏移
    当偏移过大时，调整采样策略
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        shift_threshold: float = 0.1
    ):
        """
        初始化分布偏移检测器
        
        Args:
            window_size: 滑动窗口大小
            shift_threshold: 偏移阈值
        """
        self.window_size = window_size
        self.shift_threshold = shift_threshold
        
        # 存储历史统计
        self.reward_history = deque(maxlen=window_size)
        self.value_history = deque(maxlen=window_size)
    
    def update(self, rewards: np.ndarray, values: np.ndarray):
        """
        更新历史统计
        
        Args:
            rewards: 奖励数组
            values: 价值估计数组
        """
        self.reward_history.extend(rewards)
        self.value_history.extend(values)
    
    def detect_shift(self, recent_window: int = 100) -> Dict:
        """
        检测分布偏移
        
        Args:
            recent_window: 最近窗口大小
            
        Returns:
            检测结果字典
        """
        if len(self.reward_history) < recent_window * 2:
            return {'shift_detected': False, 'shift_magnitude': 0.0}
        
        # 计算历史和最近的统计量
        recent_rewards = list(self.reward_history)[-recent_window:]
        historical_rewards = list(self.reward_history)[:-recent_window]
        
        recent_mean = np.mean(recent_rewards)
        historical_mean = np.mean(historical_rewards)
        
        recent_std = np.std(recent_rewards)
        historical_std = np.std(historical_rewards)
        
        # 计算偏移量（使用标准化的均值差异）
        if historical_std > 0:
            shift_magnitude = abs(recent_mean - historical_mean) / historical_std
        else:
            shift_magnitude = 0.0
        
        shift_detected = shift_magnitude > self.shift_threshold
        
        return {
            'shift_detected': shift_detected,
            'shift_magnitude': shift_magnitude,
            'recent_mean': recent_mean,
            'historical_mean': historical_mean,
            'recent_std': recent_std,
            'historical_std': historical_std
        }


if __name__ == "__main__":
    print("=== 测试OnPolicyAdaptiveSampler ===")
    
    sampler = OnPolicyAdaptiveSampler(buffer_size=1000)
    
    # 模拟添加轨迹
    states = np.random.randn(50, 4)
    actions = np.random.randint(0, 2, 50)
    rewards = np.random.randn(50)
    log_probs = np.random.randn(50) * 0.1
    values = np.random.randn(50)
    advantages = np.random.randn(50)
    
    sampler.add_trajectory(states, actions, rewards, log_probs, values, advantages)
    
    # 采样
    batch, weights = sampler.sample_batch(batch_size=16)
    print(f"批次大小: {len(batch['states'])}")
    print(f"重要性权重: {weights}")
    
    print("\n=== 测试GRPOAdaptiveSampler ===")
    
    grpo_sampler = GRPOAdaptiveSampler(group_size=8)
    
    # 添加一组样本
    for i in range(8):
        grpo_sampler.add_sample(
            prompt=f"prompt_{i}",
            response=f"response_{i}",
            reward=np.random.randn(),
            log_prob=np.random.randn() * 0.1
        )
    
    # 计算GRPO权重
    samples, weights = grpo_sampler.sample_with_grpo_weights()
    print(f"组大小: {len(samples)}")
    print(f"GRPO权重: {weights}")
    
    print("\n=== 测试PPOAdaptiveClipping ===")
    
    ppo_clipper = PPOAdaptiveClipping()
    
    old_log_probs = torch.randn(32)
    new_log_probs = old_log_probs + torch.randn(32) * 0.1
    advantages = torch.randn(32)
    
    loss, stats = ppo_clipper.compute_ppo_loss(old_log_probs, new_log_probs, advantages)
    print(f"PPO损失: {loss.item():.4f}")
    print(f"统计信息: {stats}")
    
    print("\n=== 测试DistributionShiftDetector ===")
    
    detector = DistributionShiftDetector()
    
    # 添加历史数据
    for _ in range(10):
        rewards = np.random.randn(100)
        values = np.random.randn(100)
        detector.update(rewards, values)
    
    # 检测偏移
    result = detector.detect_shift()
    print(f"偏移检测结果: {result}")

