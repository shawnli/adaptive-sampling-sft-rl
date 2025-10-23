# 自适应数据采样与重要性加权方法实现

本项目实现了多种基于数据重要性的自适应采样方法，用于解决SFT和RL训练中忽略不同数据重要性的问题。

## 项目背景

传统的监督微调（SFT）和强化学习（RL）训练通常采用均匀采样，忽略了不同数据样本的重要性差异。受AdaBoost等经典机器学习方法的启发，本项目实现了多种自适应采样策略，根据模型的学习效果动态调整不同数据的采样权重。

## 核心方法

### 1. Prioritized Experience Replay (PER)

**论文**: Schaul et al. (2015) "Prioritized Experience Replay"

**核心思想**:
- 根据TD误差大小计算优先级
- 使用SumTree数据结构实现O(log N)的高效采样
- 通过重要性采样权重修正非均匀采样引入的偏差

**实现文件**: `src/prioritized_replay.py`

**关键公式**:
```
优先级: p_i = |δ_i| + ε
采样概率: P(i) = p_i^α / Σ_k p_k^α
重要性权重: w_i = (N * P(i))^(-β)
```

### 2. Importance Weighted SFT (iw-SFT)

**论文**: Qin & Springenberg (2025) "Supervised Fine Tuning on Curated Data is Reinforcement Learning"

**核心思想**:
- SFT可以理解为优化RL目标的下界
- 通过重要性加权使SFT更接近RL训练
- 使用参考模型和当前模型的对数概率差计算权重

**实现文件**: `src/importance_weighted_sft.py`

**关键公式**:
```
重要性权重: w_i = exp((log π_ref(a|s) - log π_θ(a|s)) / T)
加权损失: L = Σ w_i * L_CE(y_i, ŷ_i)
```

### 3. 自适应批次采样

**核心思想**:
- 类似AdaBoost，根据样本难度动态调整权重
- 难学的样本（高损失）获得更高的采样权重
- 使用动量平滑权重更新，避免过度波动

**实现文件**: `src/prioritized_replay.py` (AdaptiveBatchSampler)

### 4. Reward加权采样

**核心思想**:
- 根据rollout的reward分布决定采样概率
- 类似Reward Weighted Regression (RWR)
- 高reward样本获得更高的采样概率

**实现文件**: `src/importance_weighted_sft.py` (RewardWeightedSampler)

## 项目结构

```
adaptive_sampling_project/
├── src/
│   ├── prioritized_replay.py          # PER和自适应采样实现
│   └── importance_weighted_sft.py     # iw-SFT和reward加权实现
├── examples/
│   └── training_example.py            # 完整训练示例
├── tests/
├── docs/
└── README.md
```

## 安装依赖

```bash
pip install torch numpy
```

## 使用示例

### 示例1: 使用PER训练DQN

```python
from src.prioritized_replay import PrioritizedReplayBuffer

# 初始化PER缓冲区
replay_buffer = PrioritizedReplayBuffer(
    capacity=10000,
    alpha=0.6,      # 优先级指数
    beta=0.4,       # 重要性采样指数
)

# 添加经验
replay_buffer.add((state, action, reward, next_state, done))

# 采样批次
batch, weights, indices = replay_buffer.sample(batch_size=32)

# 计算TD误差并更新优先级
td_errors = compute_td_errors(batch)
replay_buffer.update_priorities(indices, td_errors)
```

### 示例2: 使用iw-SFT微调语言模型

```python
from src.importance_weighted_sft import ImportanceWeightedSFTLoss

# 初始化损失函数
loss_fn = ImportanceWeightedSFTLoss(
    epsilon=0.1,
    temperature=1.0
)

# 前向传播
logits = model(input_ids)
ref_log_probs = ref_model(input_ids)  # 参考模型

# 计算iw-SFT损失
loss = loss_fn(logits, labels, ref_log_probs)
```

### 示例3: 使用自适应采样训练分类器

```python
from src.prioritized_replay import AdaptiveBatchSampler

# 初始化采样器
sampler = AdaptiveBatchSampler(
    dataset_size=1000,
    learning_rate=0.1
)

# 采样批次
indices, importance_weights = sampler.sample(batch_size=32)

# 训练后更新权重
sampler.update_weights(indices, losses)
```

## 运行完整示例

```bash
cd examples
python training_example.py
```

该示例包含三个完整的训练场景：
1. 使用PER训练DQN
2. 使用iw-SFT微调语言模型
3. 使用自适应采样训练分类器

## 核心优势

1. **提高样本效率**: 重要样本被更频繁地采样，加速学习
2. **自适应调整**: 根据模型学习状态动态调整采样策略
3. **理论保证**: 通过重要性采样修正偏差，保证收敛性
4. **易于集成**: 模块化设计，可轻松集成到现有训练流程

## 实验结果

根据原始论文报告：

- **PER**: 在49个Atari游戏中的41个超越标准DQN
- **iw-SFT**: 在AIME 2024达到66.7%，GPQA达到64.1%
- **自适应采样**: 在不平衡数据集上显著提升性能

## 参考文献

1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized Experience Replay. ICLR 2016.

2. Qin, C., & Springenberg, J. T. (2025). Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved). arXiv:2507.12856.

3. Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training Region-based Object Detectors with Online Hard Example Mining. CVPR 2016.

4. Peters, J., & Schaal, S. (2007). Reinforcement learning by reward-weighted regression for operational space control. ICML 2007.

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

