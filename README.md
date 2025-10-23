# 自适应数据采样与重要性加权方法实现

本项目实现了多种基于数据重要性的自适应采样方法，用于解决SFT和RL训练中忽略不同数据重要性的问题。**特别针对在线RL（on-policy RL）的挑战提供了专门的解决方案**。

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

**适用场景**: 离线RL、DQN等值函数方法

### 2. Importance Weighted SFT (iw-SFT)

**论文**: Qin & Springenberg (2025) "Supervised Fine Tuning on Curated Data is Reinforcement Learning"

**核心思想**:
- SFT可以理解为优化RL目标的下界
- 通过重要性加权使SFT更接近RL训练
- 使用参考模型和当前模型的对数概率差计算权重

**实现文件**: `src/importance_weighted_sft.py`

**适用场景**: LLM监督微调、高质量数据集训练

### 3. 在线RL自适应采样 ⭐ 新增

**论文**: Corrado et al. (2023) "On-Policy Policy Gradient RL Without On-Policy Sampling"

**核心思想**:
- 维护滑动窗口的经验池，保持分布新鲜度
- 基于advantage值动态调整采样权重
- 自动计算重要性采样权重修正分布偏移

**实现文件**: `src/online_rl_sampling.py`

**适用场景**: PPO、TRPO等在线策略梯度方法

**关键组件**:
- `OnPolicyAdaptiveSampler`: 在线策略自适应采样器
- `GRPOAdaptiveSampler`: GRPO（DeepSeek-R1使用的方法）
- `PPOAdaptiveClipping`: PPO自适应裁剪机制
- `DistributionShiftDetector`: 分布偏移检测器

### 4. 自适应批次采样

**核心思想**:
- 类似AdaBoost，根据样本难度动态调整权重
- 难学的样本（高损失）获得更高的采样权重
- 使用动量平滑权重更新，避免过度波动

**实现文件**: `src/prioritized_replay.py` (AdaptiveBatchSampler)

### 5. Reward加权采样

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
│   ├── importance_weighted_sft.py     # iw-SFT和reward加权实现
│   └── online_rl_sampling.py          # 在线RL自适应采样 ⭐ 新增
├── examples/
│   └── training_example.py            # 完整训练示例
├── docs/
│   ├── technical_document.md          # 详细技术文档
│   └── online_rl_analysis.md          # 在线RL分析文档 ⭐ 新增
├── tests/
└── README.md
```

## 安装依赖

```bash
pip install torch numpy
```

## 使用示例

### 示例1: 使用PER训练DQN（离线RL）

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

### 示例3: 在PPO中使用自适应采样（在线RL）⭐ 新增

```python
from src.online_rl_sampling import OnPolicyAdaptiveSampler, PPOAdaptiveClipping

# 初始化
sampler = OnPolicyAdaptiveSampler(buffer_size=10000)
ppo_clipper = PPOAdaptiveClipping()

# 训练循环
for epoch in range(num_epochs):
    # 收集轨迹
    states, actions, rewards, log_probs, values, advantages = collect_trajectories()
    
    # 添加到采样器
    sampler.add_trajectory(states, actions, rewards, log_probs, values, advantages)
    
    # 自适应采样
    batch, importance_weights = sampler.sample_batch(batch_size=256)
    
    # 计算PPO损失（带重要性采样修正）
    loss, stats = ppo_clipper.compute_ppo_loss(
        old_log_probs, new_log_probs, advantages, importance_weights
    )
    
    # 自适应调整裁剪范围
    ppo_clipper.adapt_clip_range()
```

### 示例4: 使用GRPO进行LLM RLHF ⭐ 新增

```python
from src.online_rl_sampling import GRPOAdaptiveSampler

grpo_sampler = GRPOAdaptiveSampler(group_size=8)

# 对每个prompt生成一组响应
for prompt in prompts:
    for _ in range(8):
        response = model.generate(prompt)
        reward = reward_model(prompt, response)
        log_prob = model.get_log_prob(prompt, response)
        
        grpo_sampler.add_sample(prompt, response, reward, log_prob)
    
    # 计算组内权重
    samples, weights = grpo_sampler.sample_with_grpo_weights()
    
    # 使用权重进行训练
    for sample, weight in zip(samples, weights):
        loss = compute_loss(sample) * weight
        loss.backward()
    
    grpo_sampler.clear_group()
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

## 在线RL的特殊考虑 ⚠️

在在线RL（如PPO、GRPO）中使用自适应采样需要特别注意**分布偏移问题**。我们提供了专门的解决方案：

| 挑战 | 解决方案 | 实现 |
|------|---------|------|
| 分布偏移 | 滑动窗口 + 重要性采样权重 | `OnPolicyAdaptiveSampler` |
| 双重重要性采样 | 自适应裁剪范围 | `PPOAdaptiveClipping` |
| 绝对reward问题 | 组内相对比较 | `GRPOAdaptiveSampler` |
| 偏移检测 | 实时监控统计量 | `DistributionShiftDetector` |

详细分析请参阅 [`docs/online_rl_analysis.md`](docs/online_rl_analysis.md)

## 方法对比

| 方法 | 样本效率 | 稳定性 | 计算开销 | 适用场景 |
|------|---------|--------|---------|---------|
| **PER** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 离线RL、DQN |
| **iw-SFT** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | LLM SFT |
| **OnPolicyAdaptive** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | PPO、在线RL |
| **GRPO** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | LLM RLHF |
| **AdaptiveBatch** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 监督学习 |

## 核心优势

1. **提高样本效率**: 重要样本被更频繁地采样，加速学习
2. **自适应调整**: 根据模型学习状态动态调整采样策略
3. **理论保证**: 通过重要性采样修正偏差，保证收敛性
4. **易于集成**: 模块化设计，可轻松集成到现有训练流程
5. **在线RL支持**: 专门针对在线RL的挑战提供解决方案 ⭐

## 实验结果

根据原始论文报告：

- **PER**: 在49个Atari游戏中的41个超越标准DQN
- **iw-SFT**: 在AIME 2024达到66.7%，GPQA达到64.1%
- **GRPO**: DeepSeek-R1使用该方法在多个推理任务上取得SOTA
- **自适应采样**: 在不平衡数据集上显著提升性能

## 文档

- **技术文档**: [`docs/technical_document.md`](docs/technical_document.md) - 详细的理论和实现说明
- **在线RL分析**: [`docs/online_rl_analysis.md`](docs/online_rl_analysis.md) - 在线RL中的挑战与解决方案 ⭐

## 参考文献

1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized Experience Replay. ICLR 2016.

2. Qin, C., & Springenberg, J. T. (2025). Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved). arXiv:2507.12856.

3. Corrado, N. E., & Hanna, J. P. (2023). On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling. arXiv:2311.08290. ⭐

4. Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training Region-based Object Detectors with Online Hard Example Mining. CVPR 2016.

5. Peters, J., & Schaal, S. (2007). Reinforcement learning by reward-weighted regression for operational space control. ICML 2007.

## GitHub仓库

**https://github.com/shawnli/adaptive-sampling-sft-rl**

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

- **2025-10-22**: 
  - ✅ 新增在线RL自适应采样模块 (`online_rl_sampling.py`)
  - ✅ 新增GRPO实现
  - ✅ 新增分布偏移检测器
  - ✅ 新增在线RL分析文档

