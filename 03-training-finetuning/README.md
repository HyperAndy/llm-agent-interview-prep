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

从人类偏好数据 $\mathcal{D} = \{(x, y_w, y_l)\}$ 中学习，其中 $y_w \succ y_l$ （ $y_w$ 优于 $y_l$ ）。

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
- $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ （重要性采样比）
- $\hat{A}$ ：GAE (Generalized Advantage Estimation)
- $\epsilon$ ：clip 范围（通常 0.1-0.2）

### 3.4 GAE (Generalized Advantage Estimation)

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 。

- $\lambda = 1$ ：Monte Carlo
- $\lambda = 0$ ：TD(0)
- $0 < \lambda < 1$ ：偏差-方差平衡

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

对于预训练的权重矩阵 $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ ，LoRA 添加低秩更新：

$$
\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}
$$

其中 $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$. 

**前向传播**：
$$
h = \mathbf{W}_0 x + \mathbf{B}\mathbf{A}x
$$

### 5.3 参数量分析

原始全量微调： $2 \times d \times k$ 参数（需更新 $\mathbf{W}_0$ ）

LoRA 微调： $2 \times r \times (d + k)$ 参数

**参数量 reduction**：
$$
\frac{\text{LoRA}}{\text{Full}} = \frac{2r(d+k)}{2dk} = \frac{r}{k} + \frac{r}{d} \approx \frac{r}{\min(d, k)}
$$

例如 $d=4096, k=4096, r=8$ ：仅需 $0.4\%$ 的参数！

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

冻结 $\mathbf{W}_0$, 只训练 $\mathbf{A}$ 和 $\mathbf{B}$, 参数量从 $2dk$ 降到 $2r(d+k)$. 

---

### Q5: 全参数微调需要多少显存？

**参考答案**：

粗略估算： $n$ B 参数的模型需要 $16-20 \times n$ GB 显存。

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

- $\lambda = 1$ ：Monte Carlo 估计（无偏，高方差）
- $\lambda = 0$ ：TD(0)（有偏，低方差）
- $0 < \lambda < 1$ ：偏差-方差平衡

通过调节 $\lambda$ ，可以在估计精度和稳定性之间取得平衡。

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


---

## 📝 面试高频题 & 参考答案

### Q1: LLM 训练分为哪几个阶段？各阶段的作用是什么？

**参考答案：**

| 阶段 | 目标 | 典型数据量 |
|------|------|-----------|
| **预训练（Pre-training）** | 学习通用语言能力和世界知识 | TB ~ PB 级别 |
| **有监督微调（SFT）** | 学习任务格式和指令遵循 | 万 ~ 百万级别 |
| **奖励建模（Reward Modeling）** | 学习人类偏好 / 评分能力 | 十万条级别 |
| **强化学习微调（RLHF）** | 对齐人类偏好 | 同 RM 数据 |

---

### Q2: RLHF 的全流程是什么？每一步的目标是什么？

**参考答案：**

```
预训练模型
    ↓ SFT
有监督微调模型（SFT Model）
    ↓ 训练 Reward Model
Reward Model（RM）—— 学习人类偏好
    ↓ PPO 强化学习
对齐后的最终模型
```

1. **SFT**：让模型学习"什么样的回答是好的"，建立基础的任务能力
2. **RM 训练**：给定 prompt + response，输出标量分数。训练数据是 human annotation 的偏好排序
3. **PPO 微调**：使用 RM 作为奖励信号，通过策略梯度（PPO）微调 SFT 模型，使其生成更高奖励的输出，同时用 KL 散度约束不要偏离 SFT 太远

---

### Q3: 写出 RLHF 的 PPO 目标函数，并解释每项的含义

**参考答案：**

$$
L_{	ext{PPO}} = \mathbb{E}_{(x,y)\sim\pi_{	heta_{	ext{old}}}}\left[ \min\left( r(	heta) \hat{A}, 	ext{clip}(r(	heta), 1-\epsilon, 1+\epsilon) \hat{A} 
ight) 
ight]
$$

其中：
- $r(	heta) = rac{\pi_	heta(y|x)}{\pi_{	heta_{	ext{old}}}(y|x)}$ 是新旧策略的概率比
- $\hat{A}$ 是优势函数（advantage），由 GAE 估计
- $\epsilon$ 是 clip 范围（通常 0.1~0.2），防止 $r(	heta)$ 更新过大
- KL 散度约束（ $eta \cdot 	ext{KL}[\pi_	heta || \pi_{	ext{ref}}]$ ）在完整公式中限制策略偏离参考模型

---

### Q4: 为什么需要 Reward Model？它和 SFT 模型有何区别？

**参考答案：**

- **SFT 模型**：学习"什么是好的回答"，但标注数据有限，只能覆盖有限场景
- **RM 模型**：将"回答质量"量化为一个标量分数，可以在任意 prompt-response 上打分，从而引导 RL 探索更优的回答

**RM 训练 loss（对比损失）：**

$$
L_{	ext{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) 
ight) 
ight]
$$

其中 $y_w$ 是人类偏好的回答， $y_l$ 是不偏好的回答，目标是让 $r_\phi(y_w) > r_\phi(y_l)$ 。

---

### Q5: PPO 中为什么要做 Clip？衰减因子 $\epsilon$ 的大小有何影响？

**参考答案：**

- **不 Clip**： $r(	heta)$ 可以变得很大或很小，策略更新幅度过大，训练不稳定（collapse）
- **Clip 后**：当 $r(	heta)$ 超出 $[1-\epsilon, 1+\epsilon]$ 区间时，梯度被裁断，策略不再被鼓励继续朝该方向更新
- **$\epsilon$ 大**：策略更新保守，训练慢但稳定
- **$\epsilon$ 小**：策略更新激进，可能探索更快，但容易 collapse

---

### Q6: DPO（Direct Preference Optimization）相比 RLHF 有何优势？

**参考答案：**

| 维度 | RLHF | DPO |
|------|------|-----|
| **训练流程** | 三阶段（SFT → RM → PPO） | 直接从 SFT 出发，一阶段完成 |
| **实现复杂度** | 高（需维护 RM、ref model、PPO） | 低（标准 RLHF 损失） |
| **显存占用** | 高（4 个模型） | 中（2 个模型） |
| **超参数** | KL 系数 $eta$ 、PPO $\epsilon$ 等 | 相对更少 |
| **偏好数据利用率** | 间接 | 直接 |

**DPO 损失函数：**

$$
L_{	ext{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma\left( eta \log rac{\pi_	heta(y_w|x)}{\pi_{	ext{ref}}(y_w|x)} - eta \log rac{\pi_	heta(y_l|x)}{\pi_{	ext{ref}}(y_l|x)} 
ight) 
ight]
$$

本质上是将 RM 和 PPO 的作用统一到一个二元分类损失中。

---

### Q7: LoRA 的原理是什么？为什么能加速微调？

**参考答案：**

LoRA 的核心思想：**假设权重更新 $\Delta W$ 是低秩的**，即：

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d 	imes r}, A \in \mathbb{R}^{r 	imes k}
$$

其中 $r \ll \min(d, k)$ ，训练时只优化 $A$ 和 $B$ ，而不更新 $W$ 。

**为什么能加速：**
- 训练参数量从 $2dk$ 降到 $2r(d + k)$ ，当 $r$ 很小时（如 8~16），参数量减少 90%+
- 只更新低秩矩阵 $A, B$ ，大幅减少梯度计算和优化器状态显存占用
- 推理时可以合并 $W' = W + BA$ 为等价权重，无需额外计算开销

---

### Q8: Prefix Tuning vs. LoRA vs. Adapter Tuning 的区别？

**参考答案：**

| 方法 | 可训练参数 | 推理开销 | 主要缺点 |
|------|-----------|---------|---------|
| **LoRA** | $W_A, W_B$ （低秩） | 无（可融合） | 秩选择需实验 |
| **Prefix Tuning** | 前缀 tokens embedding | 有（占序列长度） | 占用 prompt 长度，减少可用上下文 |
| **Adapter Tuning** | Adapter 层 FFN | 有（额外forward） | 推理延迟，层数多了影响明显 |
| **Full FT** | 所有参数 | 无 | 显存大、存储大 |

---

### Q9: 强化学习中 KL 散度约束的作用是什么？

**参考答案：**

KL 散度约束 $	ext{KL}[\pi_	heta || \pi_{	ext{ref}}]$ 的作用：

1. **防止过度优化**：没有 KL 约束，模型会朝着高奖励方向过度调整，生成无意义但 RM 分数高的文本（Reward Hacking）
2. **保持基本能力**：KL 散度惩罚过大的策略变化，确保模型不会在优化目标时丢失原有的语言能力
3. **平衡探索与利用**：在 RLHF 中，KL 系数 $eta$ 控制"对齐强度"——$eta$ 越大，越接近原 SFT 模型； $eta$ 越小，越激进地追求 RM 奖励

---

### Q10: 模型为什么会"胡言乱语"（幻觉）？根源是什么？

**参考答案：**

根源在于三个层面：

1. **训练数据层面**：数据存在噪声、错误或偏见，模型学到错误关联
2. **模型能力层面**：知识在预训练中被压缩存储，提取时存在不确定性
3. **优化目标层面**：NLL（Next Token Prediction）只鼓励生成"像真实文本"的序列，不区分"真实"和"虚假但像真的"

**面试加分回答**：提到 RAG、CoT、RLHF 从不同维度缓解幻觉——RAG 提供真实上下文；CoT 通过显式推理链减少跳步错误；RLHF 使模型倾向于生成"可验证为真的"内容。

---

> 📚 **参考来源**：[CSDN - 大模型面试遇到RLHF说明已成功一半](https://blog.csdn.net/2401_85327249/article/details/146503969)、[CSDN - AI大模型面试系列——微调专题](https://blog.csdn.net/m0_48891301/article/details/145322596)、[牛客网 - 字节大模型二面面经](https://www.nowcoder.com/discuss/601548478774304768)
