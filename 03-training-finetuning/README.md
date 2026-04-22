# Module 03: 训练、微调与对齐技术

## 📋 目录

1. [预训练 (Pre-training)](#1-预训练-pre-training)
2. [监督微调 (SFT)](#2-监督微调-sft)
3. [RLHF 与 PPO](#3-rlhf-与-ppo)
4. [DPO (Direct Preference Optimization)](#4-dpo-direct-preference-optimization)
5. [LoRA 微调](#5-lora-微调)
6. [显存估算与优化](#6-显存估算与优化)
7. [高频面试题](#7-高频面试题)

---

## 1. 预训练 (Pre-training)

### 1.1 预训练流程

```
┌──────────────────────────────────────────────────────────────┐
│                     Pre-training Pipeline                      │
├────────────────────────────────────────────────────────────┤
│  1. Data Collection (WebText, Wikipedia, Books, Code)         │
│                          ↓                                    │
│  2. Data Processing (Filtering, Deduplication)               │
│                          ↓                                    │
│  3. Tokenization (BPE/WordPiece/SentencePiece)                │
│                          ↓                                    │
│  4. Training (CLM/MLM, Large Batch, Long Context)            │
│                          ↓                                    │
│  5. Model Checkpoints (Every N steps)                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 预训练目标

**因果语言建模 (Causal LM)**：
$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

### 1.3 预训练关键超参数

| 参数 | GPT-3 | LLaMA |
|------|-------|-------|
| Batch Size | 3.2M tokens | 4M tokens |
| Learning Rate | $6 \times 10^{-4}$ | $1.5 \times 10^{-4}$ |
| Warmup Steps | 375 | 2000 |
| Context Length | 2048 | 2048 |
| Total Tokens | 300B | 1.4T |

---

## 2. 监督微调 (SFT)

### 2.1 SFT 流程

```
┌─────────────────────────────────────────┐
│              SFT Pipeline                 │
├─────────────────────────────────────────┤
│  Prompt (User Query)                     │
│           ↓                              │
│  Expected Response (Human-written)       │
│           ↓                              │
│  Tokenization                            │
│           ↓                              │
│  Compute Loss: CrossEntropy(predict, target)│
│           ↓                              │
│  Backprop & Update                       │
└─────────────────────────────────────────┘
```

### 2.2 SFT 损失函数

$$
\mathcal{L}_{\text{SFT}} = -\sum_{(x, y) \in \mathcal{D}} \sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x)
$$

其中 $\mathcal{D}$ 是 (prompt, response) 对的数据集。

### 2.3 SFT vs Pre-training

| 阶段 | 数据 | 目标 |
|------|------|------|
| Pre-training | 大量无标注文本 | 学习语言模型 |
| SFT | 标注的指令-响应对 | 学习遵循指令 |

---

## 3. RLHF 与 PPO

### 3.1 RLHF 三阶段流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        RLHF Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Supervised Fine-Tuning (SFT)                          │
│  "学习格式" ← 人工写的指令-响应对                                │
│                            ↓                                      │
│  Stage 2: Reward Model Training                                  │
│  "学习评分" ← 人类偏好对比数据                                    │
│  训练 Reward Model: $r_\theta(x, y)$                              │
│                            ↓                                      │
│  Stage 3: RL Fine-tuning (PPO)                                   │
│  "学习生成" ← 用 Reward Model 指导 LLM                           │
│  目标: $\max_\pi \mathbb{E}_{x \sim p, y \sim \pi(\cdot|x)} [r(x, y)]$│
│  约束: $\text{KL}(\pi \| \pi_{\text{SFT}}) \leq \delta$         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Reward Model 训练

从人类偏好数据 $\mathcal{D} = \{(x, y_w, y_l)\}$ 中学习，其中 $y_w \succ y_l$（$y_w$ 优于 $y_l$）。

**Bradley-Terry 模型**：
$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

**损失函数**：
$$
\mathcal{L}_R = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]
$$

### 3.3 PPO 算法

**强化学习目标**：
$$
\max_\theta \mathbb{E}_{x \sim p, y \sim \pi_\theta(\cdot|x)} [r(x, y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

**PPO -clip 目标**：
$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ \min\left( r(\theta) \hat{A}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A} \right) \right]
$$

其中：
- $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$（重要性采样比）
- $\hat{A}$：GAE (Generalized Advantage Estimation)
- $\epsilon$：clip 范围（通常 0.1-0.2）

### 3.4 GAE (Generalized Advantage Estimation)

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

- $\lambda = 1$：Monte Carlo
- $\lambda = 0$：TD(0)
- $0 < \lambda < 1$：偏差-方差平衡

---

## 4. DPO (Direct Preference Optimization)

### 4.1 DPO 核心思想

DPO 直接在偏好数据上优化，**无需训练 Reward Model 和使用 PPO**。

### 4.2 DPO 损失函数

$$
\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]}
$$

简化形式：
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right) \right]
$$

其中 $r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

### 4.3 DPO vs PPO

| 特性 | PPO (RLHF) | DPO |
|------|------------|-----|
| 训练阶段 | 3 阶段 (SFT → RM → PPO) | 2 阶段 (SFT → DPO) |
| 样本复杂度 | 高（需大量偏好） | 低 |
| 稳定性 | 需 KL 约束和 clipping | 更稳定 |
| 内存占用 | 高（需 critic 模型） | 低 |
| 效果 | 标准对齐 | 相当或更好 |

### 4.4 DPO 理论推导

DPO 基于 Reward Model 和最优策略的隐式关系：

**Reward 与策略的关系**（从 DPO 论文）：
$$
r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + C(x)
$$

这意味着我们可以通过比较当前策略和参考策略的对数概率比来估计 Reward。

---

## 5. LoRA 微调

### 5.1 LoRA 核心思想

LoRA (Low-Rank Adaptation) 通过**低秩矩阵分解**来高效微调大模型。

### 5.2 数学公式

对于预训练的权重矩阵 $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$，LoRA 添加低秩更新：

$$
\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}
$$

其中 $\mathbf{B} \in \mathbb{R}^{d \times r}$，$\mathbf{A} \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**前向传播**：
$$
h = \mathbf{W}_0 x + \mathbf{B}\mathbf{A}x
$$

### 5.3 参数量分析

原始全量微调：$2 \times d \times k$ 参数（需更新 $\mathbf{W}_0$）

LoRA 微调：$2 \times r \times (d + k)$ 参数

**参数量 reduction**：
$$
\frac{\text{LoRA}}{\text{Full}} = \frac{2r(d+k)}{2dk} = \frac{r}{k} + \frac{r}{d} \approx \frac{r}{\min(d, k)}
$$

例如 $d=4096, k=4096, r=8$：仅需 $0.4\%$ 的参数！

### 5.4 LoRA 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Pre-trained weight (frozen)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), 
            requires_grad=False
        )
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, x):
        # Original output
        h = F.linear(x, self.weight)
        # LoRA update (scaled by alpha/rank)
        h += (self.lora_B @ self.lora_A @ x.T).T * (self.alpha / self.rank)
        return h
```

### 5.5 LoRA 变体

| 变体 | 描述 | 特点 |
|------|------|------|
| **LoRA** | 基础低秩分解 | 通用 |
| **QLoRA** | 量化 + LoRA | 4-bit 量化，极低显存 |
| **LoRA+** | 差分学习率 | 更好的超参设置 |
| **AdaLoRA** | 自适应 rank | 根据重要性调整 r |
| **LoRA-FA** | Freezing A | 冻结 A，只训练 B |

---

## 6. 显存估算与优化

### 6.1 全量微调显存估算

对于 $n$ B 参数的模型，全量微调显存需求：

$$
\text{显存} \approx 16 \times n \text{ GB} \quad (\text{bfloat16})
$$

**详细分解**：

| 组成部分 | 显存占用 |
|----------|----------|
| 模型参数 | $2 \times n$ bytes/param (bf16) |
| 梯度 | $2 \times n$ bytes/param |
| 优化器状态 | $8 \times n$ bytes/param (Adam) |
| 激活值 | 可变（序列长度 × batch） |
| **总计** | **$\approx 20-40$ n GB** |

### 6.2 显存优化技术

| 技术 | 显存节省 | 精度影响 |
|------|----------|----------|
| **混合精度 (FP16/BF16)** | 2x | 可忽略 |
| **梯度检查点** | 60-70% | 反向传播需重新计算 |
| **梯度累积** | 变相增大 batch | 无 |
| ** ZeRO** | 显著 | 无 |
| **QLoRA** | 4x | 少量下降 |
| **CPU Offload** | 取决于内存 | 速度下降 |

### 6.3 ZeRO 优化

ZeRO (Zero Redundancy Optimizer) 将优化器状态、梯度和参数分片：

| Stage | 分片内容 | 显存节省 |
|-------|----------|----------|
| ZeRO-1 | 优化器状态 | 4x |
| ZeRO-2 | 优化器 + 梯度 | 8x |
| ZeRO-3 | 优化器 + 梯度 + 参数 | 与设备数成正比 |

### 6.4 QLoRA 量化

QLoRA 使用**NF4 (4-bit NormalFloat)** 量化：

**量化公式**：
$$
x_{\text{int4}} = \text{round}\left( \frac{x}{\Delta} \right), \quad \Delta = \frac{\text{max}(|x|)}{2^{N-1}}
$$

**NF4 优势**：
- 4-bit 量化，但保持更高精度
- 结合分页注意力（Paged Attention）
- 可在单卡 24GB 跑 65B 模型

---

## 7. 高频面试题

### Q1: RLHF 的流程是什么？

**参考答案**：

RLHF (Reinforcement Learning from Human Feedback) 包含三个阶段：

1. **SFT (监督微调)**：用人工写的指令-响应对微调预训练模型，学习输出格式
2. **Reward Model 训练**：用人类偏好对比数据训练 Reward Model
3. **PPO 微调**：用 Reward Model 作为奖励，用 PPO 算法微调 LLM

关键约束：KL 散度约束，确保微调后的策略不要偏离 SFT 太远。

---

### Q2: PPO 的 clip 机制是什么？ 为什么要 clip？

**参考答案**：

PPO 使用 clipped surrogate objective：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min\left( r(\theta) \hat{A}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A} \right) \right]
$$

**clip 的作用**：
- 当 $r(\theta) > 1+\epsilon$ 或 $r(\theta) < 1-\epsilon$ 时，停止更新
- 防止策略变化过大，保护 reward 的稳定性
- $\epsilon$ 通常取 0.1-0.2

---

### Q3: DPO 和 RLHF 的区别？

**参考答案**：

| 维度 | RLHF (PPO) | DPO |
|------|-------------|-----|
| 训练流程 | SFT → RM → PPO | SFT → DPO |
| 样本效率 | 低 | 高 |
| 显存需求 | 高（需 Critic） | 低 |
| 稳定性 | 需 KL 约束 | 更稳定 |
| 本质 | 隐式 Reward | 隐式 Reward |

DPO 直接在偏好数据上优化，省去了训练独立的 Reward Model 和 PPO 步骤。

---

### Q4: LoRA 的原理是什么？

**参考答案**：

LoRA 基于**低秩假设**：大模型的微调增量 $\Delta \mathbf{W}$ 是低秩的。

$$
\Delta \mathbf{W} = \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}, r \ll \min(d, k)
$$

**前向传播**：
$$
h = \mathbf{W}_0 x + \mathbf{B}\mathbf{A}x
$$

冻结 $\mathbf{W}_0$，只训练 $\mathbf{A}$ 和 $\mathbf{B}$，参数量从 $2dk$ 降到 $2r(d+k)$。

---

### Q5: 全参数微调需要多少显存？

**参考答案**：

粗略估算：$n$ B 参数的模型需要 $16-20 \times n$ GB 显存。

**详细计算（以 7B 模型为例）**：

| 组成部分 | 7B 模型 |
|----------|---------|
| 模型参数 (bf16) | 14 GB |
| 梯度 (bf16) | 14 GB |
| Adam 优化器 (fp32) | 56 GB |
| 激活值 (可变) | ~8-16 GB |
| **总计** | **~80-100 GB** |

需要至少 2× A100 (40GB) 或使用 ZeRO-3 + CPU Offload。

---

### Q6: 什么是 GAE？

**参考答案**：

GAE (Generalized Advantage Estimation) 是 TD(λ) 在策略梯度中的推广：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

- $\lambda = 1$：Monte Carlo 估计（无偏，高方差）
- $\lambda = 0$：TD(0)（有偏，低方差）
- $0 < \lambda < 1$：偏差-方差平衡

通过调节 $\lambda$，可以在估计精度和稳定性之间取得平衡。

---

### Q7: SFT 和 DPO 的关系是什么？

**参考答案**：

根据最新研究（复旦邱锡鹏团队），SFT 和 DPO 本质上是**隐式奖励学习的不同形式**：

1. **SFT**：通过最大化专家数据对数似然，隐式学习奖励函数
2. **DPO**：通过直接优化偏好数据，隐式学习奖励函数

两者共享同一策略-奖励最优子空间，可以互相补充。

**SFT 的问题**：KL 散度项在优化中退化为常数，缺乏有效约束。

---

### Q8: 如何选择 LoRA 的 rank？

**参考答案**：

rank 选择需要权衡：

| rank | 效果 | 参数量 | 适用场景 |
|------|------|--------|----------|
| 2-4 | 基础 | 极低 | 简单任务 |
| 8-16 | 良好 | 低 | 一般微调 |
| 32-64 | 接近全量 | 中等 | 复杂任务 |
| 128+ | 接近全量 | 较高 | 任务迁移 |

**建议**：从 rank=8 或 16 开始，根据效果调整。
