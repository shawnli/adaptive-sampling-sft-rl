"""
完整的训练示例：展示如何在实际场景中使用自适应采样方法

包括：
1. PER用于强化学习
2. iw-SFT用于语言模型微调
3. 自适应批次采样用于监督学习
"""

import sys
sys.path.append('../src')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

from prioritized_replay import PrioritizedReplayBuffer, AdaptiveBatchSampler
from importance_weighted_sft import (
    ImportanceWeightedSFTLoss,
    RewardWeightedSampler,
    AdaptiveDataWeighter
)


# ==================== 示例1: PER用于DQN训练 ====================

class SimpleDQN(nn.Module):
    """简单的DQN网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


def train_dqn_with_per():
    """使用PER训练DQN的示例"""
    print("=" * 60)
    print("示例1: 使用Prioritized Experience Replay训练DQN")
    print("=" * 60)
    
    # 超参数
    state_dim = 4
    action_dim = 2
    buffer_capacity = 10000
    batch_size = 32
    gamma = 0.99
    learning_rate = 0.001
    
    # 初始化
    policy_net = SimpleDQN(state_dim, action_dim)
    target_net = SimpleDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(
        capacity=buffer_capacity,
        alpha=0.6,
        beta=0.4
    )
    
    # 模拟收集经验
    print("\n收集初始经验...")
    for _ in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.randint(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.rand() < 0.1
        
        experience = (state, action, reward, next_state, done)
        replay_buffer.add(experience)
    
    print(f"缓冲区大小: {len(replay_buffer)}")
    
    # 训练循环
    print("\n开始训练...")
    num_updates = 100
    
    for update in range(num_updates):
        # 从PER中采样
        batch, importance_weights, indices = replay_buffer.sample(batch_size)
        
        # 解包批次
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])
        weights = torch.FloatTensor(importance_weights)
        
        # 计算Q值
        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # 计算TD误差
        td_errors = (current_q_values.squeeze() - target_q_values).detach().numpy()
        
        # 计算加权损失
        loss = (weights * (current_q_values.squeeze() - target_q_values) ** 2).mean()
        
        # 更新网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新优先级
        replay_buffer.update_priorities(indices, td_errors)
        
        if (update + 1) % 20 == 0:
            print(f"Update {update + 1}/{num_updates}, Loss: {loss.item():.4f}")
    
    print("\n训练完成！")


# ==================== 示例2: iw-SFT用于语言模型微调 ====================

class SimpleLanguageModel(nn.Module):
    """简单的语言模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits


def train_lm_with_iw_sft():
    """使用iw-SFT微调语言模型的示例"""
    print("\n" + "=" * 60)
    print("示例2: 使用Importance Weighted SFT微调语言模型")
    print("=" * 60)
    
    # 超参数
    vocab_size = 1000
    seq_length = 20
    batch_size = 16
    num_epochs = 5
    learning_rate = 0.001
    
    # 初始化模型
    model = SimpleLanguageModel(vocab_size)
    ref_model = SimpleLanguageModel(vocab_size)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = ImportanceWeightedSFTLoss(epsilon=0.1, temperature=1.0)
    
    # 创建模拟数据集（带质量分数）
    num_samples = 200
    dataset = []
    quality_scores = np.random.randn(num_samples)
    
    for i in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (seq_length,))
        labels = torch.randint(0, vocab_size, (seq_length,))
        dataset.append((input_ids, labels))
    
    # 使用reward加权采样
    sampler = RewardWeightedSampler(
        rewards=quality_scores,
        temperature=0.5
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"质量分数统计: min={quality_scores.min():.2f}, max={quality_scores.max():.2f}")
    
    # 训练循环
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # 每个epoch采样多个批次
        for _ in range(len(dataset) // batch_size):
            # 使用reward加权采样
            indices, importance_weights = sampler.sample(batch_size)
            
            # 获取批次数据
            batch_inputs = torch.stack([dataset[i][0] for i in indices])
            batch_labels = torch.stack([dataset[i][1] for i in indices])
            
            # 前向传播
            logits = model(batch_inputs)
            
            # 计算参考模型的对数概率
            with torch.no_grad():
                ref_logits = ref_model(batch_inputs)
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_log_probs = ref_log_probs.gather(
                    2, batch_labels.unsqueeze(-1)
                ).squeeze(-1)
            
            # 计算iw-SFT损失
            loss = loss_fn(logits, batch_labels, ref_log_probs)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    print("\n微调完成！")


# ==================== 示例3: 自适应批次采样用于监督学习 ====================

class SimpleClassifier(nn.Module):
    """简单的分类器"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_classifier_with_adaptive_sampling():
    """使用自适应采样训练分类器的示例"""
    print("\n" + "=" * 60)
    print("示例3: 使用自适应批次采样训练分类器")
    print("=" * 60)
    
    # 超参数
    input_dim = 20
    num_classes = 5
    num_samples = 1000
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # 创建模拟数据集
    X = np.random.randn(num_samples, input_dim)
    y = np.random.randint(0, num_classes, num_samples)
    
    # 初始化模型
    model = SimpleClassifier(input_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # 初始化自适应采样器
    adaptive_sampler = AdaptiveBatchSampler(
        dataset_size=num_samples,
        learning_rate=0.1
    )
    
    # 初始化自适应数据加权器
    data_weighter = AdaptiveDataWeighter(
        num_samples=num_samples,
        momentum=0.9,
        update_frequency=100
    )
    
    print(f"\n数据集大小: {num_samples}")
    print(f"类别数: {num_classes}")
    
    # 训练循环
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # 每个epoch采样多个批次
        for _ in range(num_samples // batch_size):
            # 使用自适应采样
            indices, importance_weights = adaptive_sampler.sample(batch_size)
            
            # 获取批次数据
            batch_X = torch.FloatTensor(X[indices])
            batch_y = torch.LongTensor(y[indices])
            weights = torch.FloatTensor(importance_weights)
            
            # 前向传播
            outputs = model(batch_X)
            losses = criterion(outputs, batch_y)
            
            # 应用重要性权重
            weighted_loss = (losses * weights).mean()
            
            # 反向传播
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            
            # 更新采样权重
            adaptive_sampler.update_weights(indices, losses.detach().numpy())
            
            # 更新数据权重
            data_weighter.update(indices, losses.detach().numpy())
            
            epoch_loss += weighted_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    print("\n训练完成！")
    
    # 分析权重分布
    final_weights = adaptive_sampler.weights
    print(f"\n最终权重统计:")
    print(f"  Min: {final_weights.min():.4f}")
    print(f"  Max: {final_weights.max():.4f}")
    print(f"  Mean: {final_weights.mean():.4f}")
    print(f"  Std: {final_weights.std():.4f}")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("自适应数据采样方法完整示例")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行示例1: PER
    train_dqn_with_per()
    
    # 运行示例2: iw-SFT
    train_lm_with_iw_sft()
    
    # 运行示例3: 自适应采样
    train_classifier_with_adaptive_sampling()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

