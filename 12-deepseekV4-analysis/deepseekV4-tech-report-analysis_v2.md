---

# DeepSeek-V4 技术报告深度解读

> **论文全称**：*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
> **发布日期**：2026年4月24日
> **技术报告**：[HuggingFace PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)

---

## 一、模型概览

DeepSeek-V4系列包含两款MoE语言模型——DeepSeek-V4-Pro（1.6T总参数，49B激活参数）和DeepSeek-V4-Flash（284B总参数，13B激活参数），均支持100万token的上下文长度。

两个模型在超过32T的多样化高质量token上进行预训练，随后经过全面的后训练流程。具体而言，V4-Pro在33T token上预训练，V4-Flash在32T token上预训练。

DeepSeek-V4通过四项协调创新来解决百万token推理问题：混合注意力架构、新的残差连接设计、不同的优化器、以及FP4量化感知训练。

### 模型参数配置表

| 模型 | 总参数 | 激活参数 | 预训练token数 | 上下文长度 | 精度 |
|------|--------|----------|-------------|-----------|------|
| V4-Pro | 1.6T | 49B | 33T | 1M | FP4+FP8混合 |
| V4-Flash | 284B | 13B | 32T | 1M | FP4+FP8混合 |

MoE专家参数使用FP4精度，其他大部分参数使用FP8精度。

---

## 二、核心创新一：混合注意力架构（CSA + HCA）

### 2.1 问题背景

标准Transformer中的原始注意力机制相对序列长度具有平方计算复杂度——将上下文翻倍约使注意力计算和内存增加四倍。在100万token时，没有架构干预则完全不可行。

### 2.2 整体设计

核心架构创新是一种混合机制，将压缩稀疏注意力（CSA）和重度压缩注意力（HCA）交替排布在Transformer层间。

CSA和HCA是具有相同目标——低成本长上下文注意力——的两种不同设计点。CSA进行轻度压缩加稀疏top-k选择；HCA进行激进压缩加稠密注意力。

### 2.3 压缩稀疏注意力（CSA）详解

CSA将每 $m$ 个token的KV缓存压缩为一个条目（使用学习的token级压缩器），然后应用DeepSeek稀疏注意力（DSA），每个query token只关注top-k个被选中的压缩KV条目。

CSA通过softmax门控池化（带有可学习位置偏置）沿序列维度将KV条目压缩4倍。一个叫做Lightning Indexer的组件负责稀疏选择，通过对query与压缩KV块的评分来实现。

**CSA的数学流程：**

**第一步：KV压缩**（压缩率 $m$）

设输入序列长度为 $n$，token的隐藏状态为 $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_n$。每 $m$ 个连续token的K和V通过softmax门控池化被压缩为一个条目：

$$\tilde{\mathbf{k}}_j = \sum_{i=(j-1)m+1}^{jm} \alpha_i \cdot \mathbf{k}_i, \qquad \tilde{\mathbf{v}}_j = \sum_{i=(j-1)m+1}^{jm} \alpha_i \cdot \mathbf{v}_i$$

其中 $\alpha_i$ 来自softmax门控（含可学习位置偏置）：

$$\alpha_i = \frac{\exp(s_i + b_i)}{\sum_{i'=(j-1)m+1}^{jm} \exp(s_{i'} + b_{i'})}$$

压缩后，原始 $n$ 个KV条目变为 $\lceil n/m \rceil$ 个压缩条目。

**第二步：Lightning Indexer 稀疏选择**

Lightning Indexer在CSA内部以FP4精度运行。它对每个query token打分所有压缩KV块，选出top-k个最相关的：

$$\text{score}_j = \text{ReLU}\left(\mathbf{q} \cdot \tilde{\mathbf{k}}_j^{\top}\right)$$

$$\mathcal{S} = \text{top-k}\left(\{\text{score}_j\}_{j=1}^{\lceil n/m \rceil}\right)$$

V4-Pro选择top 1024个，V4-Flash选择top 512个压缩KV条目。

**第三步：核心注意力计算**

仅在选出的 $\mathcal{S}$ 集合上做Multi-Query Attention：

$$\text{Attn}(\mathbf{q}, \mathcal{S}) = \text{softmax}\left(\frac{\mathbf{q} \cdot \tilde{\mathbf{K}}_\mathcal{S}^\top}{\sqrt{d_k}}\right) \tilde{\mathbf{V}}_\mathcal{S}$$

**CSA流程图：**

```
输入序列: [t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈, ..., t_n]  (n个token)
                    │
                    ▼
    ┌─────────────────────────────────────────┐
    │  Step 1: KV压缩 (m=4)                   │
    │  每4个token → 1个压缩KV条目              │
    │  [t₁t₂t₃t₄]→c̃₁  [t₅t₆t₇t₈]→c̃₂  ...  │
    │  n个token → n/4 个压缩条目               │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │  Step 2: Lightning Indexer (FP4)         │
    │  对每个query，给所有 n/4 个压缩块打分    │
    │  选出 top-k 个最相关的压缩块             │
    │  Pro: k=1024, Flash: k=512              │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │  Step 3: 稀疏注意力                      │
    │  仅在 k 个选中的压缩KV上做MQA           │
    │  + 滑动窗口(128 tokens)本地注意力        │
    └─────────────────────────────────────────┘
```

### 2.4 重度压缩注意力（HCA）详解

HCA更加激进：它将每 $m'$ 个token的KV条目整合为一个压缩条目——其中 $m' \gg m$——然后对这些表示进行稠密注意力。

在DeepSeek-V4中，$m'$ 被设置为128。所以不是将4个token合并成一个条目，HCA将128个token合并成一个条目。

这样，100万token只剩约7,800张"摘要卡片"。由于卡片数已经非常少，所以每次全部读取，不需要top-k选择。

**HCA的数学流程：**

$$\hat{\mathbf{k}}_j = \text{Compress}_{m'}\left(\mathbf{k}_{(j-1)m'+1}, \dots, \mathbf{k}_{jm'}\right)$$

$$\hat{\mathbf{v}}_j = \text{Compress}_{m'}\left(\mathbf{v}_{(j-1)m'+1}, \dots, \mathbf{v}_{jm'}\right)$$

压缩后序列长度：$n / m' = 1{,}000{,}000 / 128 \approx 7{,}813$

然后对所有压缩条目做**稠密注意力**（不做稀疏选择）：

$$\text{HCA}(\mathbf{q}) = \text{softmax}\left(\frac{\mathbf{q} \cdot \hat{\mathbf{K}}^\top}{\sqrt{d_k}}\right) \hat{\mathbf{V}}$$

**HCA流程图：**

```
输入序列: [t₁, t₂, ..., t₁₂₈, t₁₂₉, ..., t₂₅₆, ..., t_1M]
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │  重压缩 (m'=128)                         │
    │  每128个token → 1个压缩KV条目            │
    │  1M token → ~7,813 个压缩条目            │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  稠密注意力                               │
    │  每个query关注所有 ~7,813 个压缩条目     │
    │  无需稀疏选择 (序列已足够短)              │
    │  + 滑动窗口(128 tokens)本地注意力         │
    └──────────────────────────────────────────┘
```

### 2.5 层间交替策略与辅助机制

在DeepSeek-V4中，层是交替排布的。V4-Flash前两层使用纯滑动窗口注意力，其余层在CSA和HCA之间交替。V4-Pro前两层使用HCA，其余层在CSA和HCA之间交替。

在V4-Pro的61层堆叠中，第0-1层为HCA，第2-60层在CSA和HCA之间交替，末端的MTP块仅运行滑动窗口。

**滑动窗口分支：**
CSA和HCA都包含一个滑动窗口注意力分支，覆盖最近的 $n_{\text{win}}$ 个token以建模局部依赖。
窗口大小为128个token。

**Attention Sink（注意力沉坑）：**
在CSA和HCA的核心注意力中，DeepSeek-V4向softmax分母添加一小组可学习的sink logits。这允许每个注意力头的分数总和小于1。在标准注意力中，分数被强制恰好求和为1。这意味着每个query token必须将所有注意力分配到某处，即使上下文中没有真正相关的内容。Attention sinks让模型能够表达"当前没有重要的东西"并减少总体注意力。

**QK归一化：**
DeepSeek-V4还在核心注意力步骤之前对query和压缩KV条目应用RMSNorm，以保持注意力logits稳定并防止其爆炸。

### 2.6 整体架构图

```
┌══════════════════════════════════════════════════════════════════┐
║                    DeepSeek-V4 整体架构                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  输入 Embedding (128K Vocabulary)                                ║
║       │                                                          ║
║       ▼                                                          ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │  Layer 0-1: HCA层 (V4-Pro) / 滑动窗口(V4-Flash)           │  ║
║  └────────────────────┬───────────────────────────────────────┘  ║
║       │               │                                          ║
║       ▼               │ mHC残差                                  ║
║  ┌────────────┐  ┌────┴───────┐  ┌────────────┐                 ║
║  │ Layer 2:   │  │ Layer 3:   │  │ Layer 4:   │                 ║
║  │   CSA      │→ │   HCA      │→ │   CSA      │→  ...          ║
║  │ ┌────────┐ │  │ ┌────────┐ │  │ ┌────────┐ │                 ║
║  │ │压缩m=4 │ │  │ │压缩m=128│ │  │ │压缩m=4 │ │                 ║
║  │ │top-k   │ │  │ │稠密attn │ │  │ │top-k   │ │                 ║
║  │ │+滑动窗口│ │  │ │+滑动窗口│ │  │ │+滑动窗口│ │                 ║
║  │ │+sink   │ │  │ │+sink   │ │  │ │+sink   │ │                 ║
║  │ └────────┘ │  │ └────────┘ │  │ └────────┘ │                 ║
║  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                 ║
║        │ mHC           │ mHC           │ mHC                     ║
║        ▼               ▼               ▼                         ║
║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                ║
║  │ DeepSeekMoE │ │ DeepSeekMoE │ │ DeepSeekMoE │                ║
║  │ (FFN层)     │ │ (FFN层)     │ │ (FFN层)     │                ║
║  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │                ║
║  │ │共享专家  │ │ │ │共享专家  │ │ │ │共享专家  │ │                ║
║  │ │+路由专家 │ │ │ │+路由专家 │ │ │ │+路由专家 │ │                ║
║  │ │(FP4权重) │ │ │ │(FP4权重) │ │ │ │(FP4权重) │ │                ║
║  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │                ║
║  └─────┬───────┘ └─────┬───────┘ └─────┬───────┘                ║
║        │ mHC           │ mHC           │ mHC                     ║
║        ▼               ▼               ▼                         ║
║       ...  (共61层, V4-Pro)  ...                                 ║
║       │                                                          ║
║       ▼                                                          ║
║  MTP Block (滑动窗口注意力)                                       ║
║       │                                                          ║
║       ▼                                                          ║
║  输出层 (Language Model Head)                                     ║
╚══════════════════════════════════════════════════════════════════╝
```

### 2.7 效率数据

在100万token上下文设置下，DeepSeek-V4-Pro仅需DeepSeek-V3.2的27%的单token推理FLOPs和10%的KV缓存。

V4-Flash的降幅更大，仅使用10%的计算力和7%的内存。

| 指标 | V4-Pro vs V3.2 | V4-Flash vs V3.2 |
|------|:---:|:---:|
| 单token推理FLOPs | **27%** ↓73% | **10%** ↓90% |
| KV缓存大小 | **10%** ↓90% | **7%** ↓93% |

---

## 三、核心创新二：流形约束超连接（mHC）

### 3.1 从残差连接到超连接的演进

在普通Transformer中，每一层将其输出添加到"残差流"中——一个从模型底部流向顶部的共享信号。超连接（HC）将这个残差流扩展 $n_{\text{hc}}$ 倍（在DeepSeek-V4中设为4），将单一流变为多个并行流。三个可学习矩阵（A, B, C）在每一层混合这些流，赋予模型更大的信息流动灵活性。

### 3.2 HC的问题

普通HC的问题在于，当堆叠许多层时会变得数值不稳定。信号可能爆炸或坍缩，训练会崩溃。

未约束的超连接在DeepSeek自己的27B实验中导致信号放大超过3000倍，引发灾难性训练发散。

### 3.3 mHC的数学原理

mHC的状态更新公式为：

$$\mathbf{X}_{l+1} = \mathbf{B}_l \mathbf{X}_l + \mathbf{C}_l \mathcal{F}_l(\mathbf{A}_l \mathbf{X}_l)$$

其中：
- $\mathbf{X}_l \in \mathbb{R}^{n_{\text{hc}} \times d}$：第 $l$ 层之前的残差状态（$n_{\text{hc}}=4$ 条并行流）
- $\mathcal{F}_l$：非线性变换（如MoE层）
- $\mathbf{A}_l$：输入映射矩阵（将多流聚合为层输入）
- $\mathbf{C}_l$：输出映射矩阵（将层输出分发回多流）
- $\mathbf{B}_l$：**残差映射矩阵**（关键创新所在）

mHC将残差映射矩阵约束在Birkhoff多面体——双随机矩阵集合——这一特殊流形上。双随机矩阵是每行和每列之和均为1的矩阵。

**双随机矩阵的定义：**

$$\mathbf{B}_l \in \mathcal{DS}_{n_{\text{hc}}} = \left\{ \mathbf{M} \in \mathbb{R}^{n \times n}_{\geq 0} \;\middle|\; \sum_j M_{ij} = 1 \;\forall i, \;\; \sum_i M_{ij} = 1 \;\forall j \right\}$$

**三个关键数学性质：**

**1. 范数保持**：谱范数被限制在1以内，因此梯度不会爆炸。**2. 组合封闭性**：双随机矩阵的乘积仍然是双随机矩阵；因此整个网络深度仍然稳定。

$$\|\mathbf{B}_l\|_2 \leq 1 \implies \left\|\prod_{l=1}^{L} \mathbf{B}_l\right\|_2 \leq 1$$

在几何层面，双随机矩阵的集合构成Birkhoff多面体，它是所有置换矩阵的凸包。每个双随机矩阵都可以写成置换矩阵的加权平均。置换只是重排，不会放大信号。重排的加权平均也不会放大。结果：无论网络多深，复合增益保持接近1。

### 3.4 Sinkhorn-Knopp投影算法

Sinkhorn-Knopp算法是一种迭代方法，交替归一化行和列直到达到所需精度。实验中确定20次迭代能在不过度计算的情况下得到良好近似。

```
算法：Sinkhorn-Knopp 投影
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 未约束的残差映射矩阵 B̂_l ∈ ℝⁿˣⁿ
输出: 双随机矩阵 B_l ∈ DS_n

1. B ← exp(B̂_l)       // 确保所有元素为正
2. for t = 1 to 20:
3.     B ← B / (B · 1)   // 行归一化: 每行除以行和
4.     B ← B / (1ᵀ · B)  // 列归一化: 每列除以列和
5. return B_l ← B
```

### 3.5 实验效果

mHC使信号放大倍数从3000x降至1.6x，使得在1.6万亿参数规模下的稳定训练成为可能。

在27B模型上的基准结果显示mHC在所有任务上都优于基线和HC。BBH从43.8（基线）提升到48.9（HC）再到51.0（mHC）。DROP、GSM8K、MMLU等也呈类似模式。

4倍宽的残差流仅增加6.7%的训练时间开销。

```
信号放大增益对比（64层深度）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
标准残差连接:    ≈ 1.0x    ✓ 稳定但表达力不足
HC (未约束):    > 3000x   ✗ 训练发散崩溃！
mHC (约束):     ≈ 1.6x    ✓ 稳定且表达力强
```

---

## 四、核心创新三：Muon优化器

### 4.1 基本原理

DeepSeek-V4将大部分参数的优化器从AdamW换成了Muon。Muon是一种使用完整梯度矩阵并正交化更新方向的优化器。

Muon查看完整梯度矩阵并在权重更新之前应用正交化步骤。这个正交化通过Newton-Schulz迭代完成。它将更新矩阵的所有奇异值归一化到接近1，因此没有任何单一方向在更新中占主导。结果是更快的收敛和更稳定的训练。

### 4.2 Hybrid Newton-Schulz迭代

Muon使用Newton-Schulz迭代来近似正交化梯度更新矩阵。实现使用混合两阶段调度：8次迭代使用系数 $(3.4445, -4.7750, 2.0315)$ 以快速收敛，然后2次稳定迭代使用系数 $(2, -1.5, 0.5)$。

**Newton-Schulz迭代公式：**

给定梯度矩阵 $\mathbf{G} \in \mathbb{R}^{m \times n}$，设 $\mathbf{X}_0 = \mathbf{G} / \|\mathbf{G}\|_F$，使用多项式加速的迭代：

$$\mathbf{X}_{k+1} = a_k \mathbf{X}_k + b_k (\mathbf{X}_k \mathbf{X}_k^\top)\mathbf{X}_k + c_k (\mathbf{X}_k \mathbf{X}_k^\top)^2 \mathbf{X}_k$$

**第一阶段**（8次，快速收敛）：$a=3.4445, \; b=-4.7750, \; c=2.0315$

**第二阶段**（2次，稳定化）：$a=2, \; b=-1.5, \; c=0.5$

收敛后，$\mathbf{X}_{10}$ 近似为 $\mathbf{G}$ 的**极分解**中的正交因子 $\mathbf{U}$：

$$\mathbf{G} = \mathbf{U} \mathbf{P}, \quad \mathbf{U}^\top \mathbf{U} = \mathbf{I}$$

**最终权重更新（含Nesterov动量）：**

$$\boldsymbol{\mu}_{t} = \beta \boldsymbol{\mu}_{t-1} + (1-\beta) \mathbf{G}_t$$

$$\hat{\boldsymbol{\mu}}_{t} = \beta \boldsymbol{\mu}_{t} + (1-\beta) \mathbf{G}_t \quad \text{(Nesterov look-ahead)}$$

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \cdot \frac{\text{NS-Ortho}(\hat{\boldsymbol{\mu}}_{t})}{\text{RMS}(\text{NS-Ortho}(\hat{\boldsymbol{\mu}}_{t}))}$$

### 4.3 参数适用范围

AdamW保留给embedding模块、预测头、mHC模块的静态偏置和门控因子、以及所有RMSNorm权重。其他所有参数使用Muon。

在1.6万亿参数规模下，训练不稳定性迅速复合。Muon与mHC的稳定性保证相结合，使DeepSeek能够将训练推进到33万亿token，而不会出现通常困扰该规模模型的梯度坍缩。

---

## 五、核心创新四：FP4量化感知训练

为了部署效率，FP4 (MXFP4) 量化感知训练 (QAT) 被应用于MoE专家权重和CSA中Lightning Indexer的QK路径。

在推理和RL rollout期间，直接使用真正的FP4权重（而非模拟量化），减少内存流量和采样延迟。

在训练时，FP32主权重被量化为FP4并无损反量化为FP8用于计算，使模型能原生适应低精度推理。

**训练-推理精度流程：**

```
训练时:
  FP32 主权重 ──→ 量化为 FP4 ──→ 反量化为 FP8 ──→ 前向/反向传播
                                                      │
                                                      ▼
                                                  FP32 梯度更新

推理/RL时:
  FP4 权重 ──→ 直接用于计算 ──→ 降低内存流量 + 加速推理
```

---

## 六、MoE架构改进

### 6.1 路由门控函数改进

专家亲和度激活函数从标准的Sigmoid变更为数学上截然不同的 $\sqrt{\text{Softplus}(\cdot)}$。

$$\text{V3门控: } g(x) = \sigma(x) = \frac{1}{1+e^{-x}}$$

$$\text{V4门控: } g(x) = \sqrt{\text{Softplus}(x)} = \sqrt{\ln(1+e^x)}$$

### 6.2 Hash路由MoE替换初始Dense FFN

V4从V3继承了MoE和MTP框架，但积极重构了路由机制。初始的Dense FFN层被完全替换为Hash路由的MoE层。

### 6.3 V4-Flash MoE配置

V4-Flash的MoE配置：1个共享专家 + 256个路由专家，每token激活6个，专家hidden dim为2048。

---

## 七、训练稳定性技巧

训练万亿参数MoE模型引入了显著的不稳定性。两种技术被证明有效：

### 7.1 Anticipatory Routing（预见性路由）

预见性路由将主干网络和路由网络的更新解耦：步骤 $t$ 的路由索引使用历史参数 $\theta_{t-\Delta t}$ 计算，打破了路由决策强化MoE层中异常值的循环。

$$\text{路由索引}_t = \text{Router}(\mathbf{x}_t; \theta_{t-\Delta t}) \quad \text{而非} \quad \text{Router}(\mathbf{x}_t; \theta_t)$$

### 7.2 SwiGLU Clamping（SwiGLU截断）

SwiGLU Clamping将SwiGLU的线性分量约束在 $[-10, 10]$，并将门控分量的上界钳制在10，直接抑制异常激活。

$$\text{SwiGLU}(x) = \text{Swish}(\text{clamp}(\mathbf{W}_g x, -10, 10)) \odot \text{clamp}(\mathbf{W}_1 x, -10, 10)$$

预见性路由和SwiGLU Clamping在实践中有效，但团队承认其"底层原理仍未被充分理解"，并将训练稳定性列为基础研究的活跃领域。

---

## 八、后训练流程：On-Policy Distillation

后训练流程用On-Policy Distillation（OPD）替代了DeepSeek-V3.2的混合RL阶段。独立的领域专家首先在数学、编码、Agent任务和指令遵循等方面通过SFT和基于GRPO的强化学习进行训练。

不同于在通用模型上运行一个大型混合RL阶段，DeepSeek为每个领域单独训练专家（数学、编码、Agent任务、指令遵循等）。

**OPD流程图：**

```
阶段一：独立领域专家训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 数学专家  │  │ 编码专家  │  │ Agent专家 │  │ 指令专家  │
│          │  │          │  │          │  │          │
│ SFT →RL  │  │ SFT →RL  │  │ SFT →RL  │  │ SFT →RL  │
│ (GRPO)   │  │ (GRPO)   │  │ (GRPO)   │  │ (GRPO)   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     └──────┬──────┴─────────────┴──────┬──────┘
            │                           │
            ▼                           ▼
阶段二：On-Policy Distillation (OPD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────┐
│            学生模型（统一模型）               │
│                                             │
│  1. 学生模型生成样本 (on-policy)             │
│  2. 多个教师专家对样本输出 logits            │
│  3. 全词表 KL 散度蒸馏                       │
│     L = KL(p_teacher || p_student)          │
│  4. 仅缓存最后一层隐藏状态（非完整logits）   │
│  5. 训练时按需重建logits                     │
│  6. TileLang专用kernel加速KL计算             │
└─────────────────────────────────────────────┘
```

---

## 九、Agentic能力与推理模式

### 9.1 三种推理模式

每个V4模型暴露三种推理努力级别：Non-Think、High和Max。

推理模式的影响远大于原始规模对困难推理任务的影响：HLE从7.7（Pro Non-Think）飙升至37.7（Pro Max），接近5倍的跳跃。

### 9.2 Agentic基础设施

DeepSeek Elastic Compute (DSec) 是一个Rust平台，在一个Python SDK后暴露四种执行基底：函数调用、容器、microVM（Firecracker）和完整VM（QEMU）。单个集群运行数十万个并发沙箱。

---

## 十、工程系统创新

### 10.1 KV缓存存储

两种路径均使用FP8存储大部分KV条目，仅对RoPE维度使用BF16。

### 10.2 TileLang与专用内核

工程工作也很可观：融合MoE内核、TileLang内核开发、FP4量化感知训练、批量不变确定性内核、长上下文注意力的两阶段上下文并行、以及带磁盘存储的异构KV缓存系统。

### 10.3 vLLM支持

该模型使用c4a和c128a注意力的混合，一些注意力层纯粹使用滑动窗口进行本地信息处理而不进行压缩。

在DeepSeek V4中，有两种压缩方式：c4a将KV缓存压缩约1/4，一个压缩token是8个未压缩token的加权和，步幅为4。c128a将KV缓存压缩约1/128，一个压缩token是128个未压缩token的加权和，步幅为128。

---

## 十一、性能基准

V4-Pro-Max在Codeforces上达到3206评分，领先GPT-5.4-xHigh（3168）和Gemini-3.1-Pro-High（3052）。在SimpleQA Verified上得分57.9 Pass@1，超过Claude Opus 4.6 Max（46.2）和GPT-5.4-xHigh（45.3）。在SWE-Verified上达到80.6% resolved，与Gemini-3.1-Pro-High（80.6%）持平，仅略低于Claude Opus 4.6 Max（80.8%）。

在长上下文基准上，V4-Pro-Max在OpenAI MRCR 1M上得分83.5 MMR，在CorpusQA 1M上得分62.0准确率，超过Gemini-3.1-Pro-High（分别为76.3和53.8），但在两者上都落后于Claude Opus 4.6 Max（92.9和71.7）。

| 基准测试 | V4-Pro-Max | Claude Opus 4.6 | GPT-5.4 | Gemini 3.1 Pro |
|---------|:---:|:---:|:---:|:---:|
| **Codeforces** | **3206** | - | 3168 | 3052 |
| **LiveCodeBench** | **93.5** | 88.8 | - | 91.7 |
| **SWE-Verified** | 80.6 | **80.8** | - | 80.6 |
| **SimpleQA-V** | **57.9** | 46.2 | 45.3 | 75.6 |
| **MRCR 1M** | 83.5 | **92.9** | - | 76.3 |
| **CorpusQA 1M** | 62.0 | **71.7** | - | 53.8 |

---

## 十二、创新技术总览

```
┌══════════════════════════════════════════════════════════════════════════════┐
║                    DeepSeek-V4 创新技术全景图                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─ 架构创新 ──────────────────────────────────────────────────────────┐    ║
║  │  ① CSA (压缩稀疏注意力)     m=4 压缩 + top-k 稀疏 + 滑动窗口       │    ║
║  │  ② HCA (重度压缩注意力)     m'=128 重压缩 + 稠密注意力 + 滑动窗口   │    ║
║  │  ③ mHC (流形约束超连接)     Birkhoff多面体 + Sinkhorn-Knopp投影      │    ║
║  │  ④ MoE改进                 √Softplus门控 + Hash路由 + FP4专家       │    ║
║  │  ⑤ Attention Sink          可学习logits, 允许注意力和<1             │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌─ 训练创新 ──────────────────────────────────────────────────────────┐    ║
║  │  ⑥ Muon优化器              Newton-Schulz正交化 + Nesterov动量       │    ║
║  │  ⑦ FP4 QAT                量化感知训练, 推理直接用FP4               │    ║
║  │  ⑧ Anticipatory Routing   用历史参数θ_{t-Δt}解耦路由更新           │    ║
║  │  ⑨ SwiGLU Clamping        线性[-10,10] + 门控上界10                │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌─ 后训练创新 ────────────────────────────────────────────────────────┐    ║
║  │  ⑩ On-Policy Distillation  独立领域专家 → 全词表logit蒸馏          │    ║
║  │  ⑪ 三级推理模式            Non-Think / High / Max                  │    ║
║  │  ⑫ DSec沙箱                Rust平台, 数十万并发沙箱用于Agent RL     │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌─ 系统工程 ──────────────────────────────────────────────────────────┐    ║
║  │  ⑬ MegaMoE                 通信-计算融合内核, 加速1.5~1.96x        │    ║
║  │  ⑭ TileLang                DSL内核开发 + Z3 SMT形式化验证           │    ║
║  │  ⑮ 批量不变性/确定性       每次运行梯度累积顺序一致                 │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 十三、V3到V4的关键变化总结

| 维度 | DeepSeek-V3/V3.2 | DeepSeek-V4 |
|------|:---:|:---:|
| 注意力 | MLA (Multi-head Latent Attention) | CSA + HCA混合注意力 |
| 残差连接 | 标准残差 | mHC (Birkhoff多面体约束) |
| 优化器 | AdamW | Muon (Newton-Schulz正交化) |
| 精度 | FP8 | FP4+FP8混合 (QAT) |
| 上下文长度 | 128K | **1M** (8x扩展) |
| 后训练 | 混合RL | On-Policy Distillation |
| 初始FFN层 | Dense FFN | Hash路由MoE |
| 门控函数 | Sigmoid | √Softplus |
| 训练稳定 | - | Anticipatory Routing + SwiGLU Clamping |

---

为降低最大架构变更（mHC、混合CSA和HCA、Muon、FP4）的风险，V4保留了许多先前验证过的V3组件。DeepSeek将最终架构描述为"相对复杂"，并表示未来版本将尝试"将架构精炼到最本质的设计"。

