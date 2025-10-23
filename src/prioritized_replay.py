"""
Prioritized Experience Replay (PER) Implementation
基于TD误差的优先级经验回放，用于强化学习中的自适应数据采样

参考论文: Schaul et al. (2015) "Prioritized Experience Replay"
"""

import numpy as np
import torch
from typing import Tuple, List, Optional


class SumTree:
    """
    SumTree数据结构，用于高效的优先级采样
    
    使用完全二叉树的数组表示，父节点存储子节点的和
    时间复杂度: O(log N) 更新和采样
    """
    
    def __init__(self, capacity: int):
        """
        初始化SumTree
        
        Args:
            capacity: 最大容量
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级和
        self.data = np.zeros(capacity, dtype=object)  # 存储实际数据
        self.write_idx = 0
        self.size = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def update(self, idx: int, priority: float):
        """
        更新叶节点的优先级
        
        Args:
            idx: 数据索引
            priority: 新的优先级值
        """
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def add(self, priority: float, data):
        """
        添加新数据
        
        Args:
            priority: 优先级
            data: 要存储的数据
        """
        self.data[self.write_idx] = data
        self.update(self.write_idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self, cumsum: float) -> Tuple[int, float, object]:
        """
        根据累积和检索数据
        
        Args:
            cumsum: 累积和值
            
        Returns:
            (数据索引, 优先级, 数据)
        """
        idx = 0
        
        while True:
            left = 2 * idx + 1
            right = left + 1
            
            if left >= len(self.tree):
                break
            
            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = right
        
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx], self.data[data_idx]
    
    @property
    def total_priority(self) -> float:
        """返回总优先级"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    
    根据TD误差动态调整样本的采样概率，重要的经验被更频繁地采样
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        初始化优先级回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数，控制优先级的使用程度 (0=均匀采样, 1=完全优先级)
            beta: 重要性采样权重指数，用于修正偏差 (0=无修正, 1=完全修正)
            beta_increment: beta的增量，随训练逐渐增加
            epsilon: 最小优先级，防止优先级为0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
    
    def add(self, experience):
        """
        添加经验，使用最大优先级
        
        Args:
            experience: 经验元组 (state, action, reward, next_state, done)
        """
        priority = self.max_priority
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, List[int]]:
        """
        采样一个批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (经验列表, 重要性采样权重, 树索引列表)
        """
        batch = []
        indices = []
        priorities = []
        
        # 将总优先级范围分成batch_size个段
        segment = self.tree.total_priority / batch_size
        
        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # 从每个段中均匀采样
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(value)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # 计算采样概率
        priorities = np.array(priorities)
        probs = priorities / self.tree.total_priority
        
        # 计算重要性采样权重
        # w_i = (N * P(i))^(-beta)
        weights = (self.tree.size * probs) ** (-self.beta)
        
        # 归一化权重，使最大权重为1
        weights = weights / weights.max()
        
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        根据TD误差更新优先级
        
        Args:
            indices: 要更新的索引列表
            td_errors: TD误差数组
        """
        for idx, td_error in zip(indices, td_errors):
            # p_i = |δ_i| + ε
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.size


class AdaptiveBatchSampler:
    """
    自适应批次采样器
    
    根据样本的学习难度动态调整采样概率
    类似于AdaBoost的思想，难学的样本获得更高的权重
    """
    
    def __init__(
        self,
        dataset_size: int,
        initial_weight: float = 1.0,
        learning_rate: float = 0.1
    ):
        """
        初始化自适应采样器
        
        Args:
            dataset_size: 数据集大小
            initial_weight: 初始权重
            learning_rate: 权重更新的学习率
        """
        self.weights = np.ones(dataset_size) * initial_weight
        self.learning_rate = learning_rate
        self.dataset_size = dataset_size
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据权重采样批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (采样索引, 重要性权重)
        """
        # 归一化权重得到概率
        probs = self.weights / self.weights.sum()
        
        # 根据概率采样
        indices = np.random.choice(
            self.dataset_size,
            size=batch_size,
            replace=False,
            p=probs
        )
        
        # 计算重要性采样权重
        sampled_probs = probs[indices]
        importance_weights = 1.0 / (self.dataset_size * sampled_probs)
        importance_weights = importance_weights / importance_weights.max()
        
        return indices, importance_weights
    
    def update_weights(self, indices: np.ndarray, losses: np.ndarray):
        """
        根据损失更新样本权重
        
        Args:
            indices: 样本索引
            losses: 对应的损失值
        """
        # 归一化损失到[0, 1]
        if losses.max() > losses.min():
            normalized_losses = (losses - losses.min()) / (losses.max() - losses.min())
        else:
            normalized_losses = np.ones_like(losses)
        
        # 更新权重：损失越大，权重增加越多
        for idx, loss in zip(indices, normalized_losses):
            self.weights[idx] *= np.exp(self.learning_rate * loss)
        
        # 防止权重过大或过小
        self.weights = np.clip(self.weights, 1e-6, 1e6)


if __name__ == "__main__":
    # 测试SumTree
    print("=== 测试SumTree ===")
    tree = SumTree(capacity=5)
    
    for i in range(5):
        tree.add(priority=i+1, data=f"data_{i}")
    
    print(f"总优先级: {tree.total_priority}")
    
    # 测试采样
    idx, priority, data = tree.get(7.5)
    print(f"采样结果: idx={idx}, priority={priority}, data={data}")
    
    # 测试优先级回放缓冲区
    print("\n=== 测试PrioritizedReplayBuffer ===")
    buffer = PrioritizedReplayBuffer(capacity=100)
    
    # 添加一些经验
    for i in range(50):
        experience = (i, i*2, i*0.1, i+1, False)
        buffer.add(experience)
    
    # 采样
    batch, weights, indices = buffer.sample(batch_size=8)
    print(f"采样批次大小: {len(batch)}")
    print(f"重要性权重: {weights}")
    
    # 更新优先级
    td_errors = np.random.rand(8)
    buffer.update_priorities(indices, td_errors)
    print(f"已更新优先级")
    
    # 测试自适应批次采样器
    print("\n=== 测试AdaptiveBatchSampler ===")
    sampler = AdaptiveBatchSampler(dataset_size=100)
    
    indices, weights = sampler.sample(batch_size=10)
    print(f"采样索引: {indices}")
    print(f"重要性权重: {weights}")
    
    # 模拟损失并更新权重
    losses = np.random.rand(10)
    sampler.update_weights(indices, losses)
    print(f"已更新样本权重")

