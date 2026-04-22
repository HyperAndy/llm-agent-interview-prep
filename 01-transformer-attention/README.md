# Module 01: Transformer 与 Attention 机制

## 📋 目录

1. [Transformer 整体架构](#1-transformer-整体架构)
2. [Self-Attention 机制](#2-self-attention-机制)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [位置编码 (Positional Encoding)](#4-位置编码-positional-encoding)
5. [Layer Normalization](#5-layer-normalization)
6. [前馈神经网络 (FFN)](#6-前馈神经网络-ffn)
7. [Encoder-Decoder vs Decoder-Only](#7-encoder-decoder-vs-decoder-only)
8. [高频面试题](#8-高频面试题)

---

## 1. Transformer 整体架构

### 1.1 标准 Transformer 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Transformer Encoder                     │
├─────────────────────────────────────────────────────────────┤
│  Input Embedding + Positional Encoding                       │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Encoder Layer (×N)                      │     │
│  │  ┌─────────────────┐    ┌─────────────────┐        │     │
│  │  │  Multi-Head     │    │     FFN         │        │     │
│  │  │  Self-Attention │    │  (Feed-Forward) │        │     │
│  │  │                 │    │                 │        │     │
│  │  │  Add & Norm     │    │  Add & Norm     │        │     │
│  │  └─────────────────┘    └─────────────────┘        │     │
│  └─────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│              Encoder Output (Hidden States)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Transformer Decoder                     │
├─────────────────────────────────────────────────────────────┤
│  Output Embedding + Positional Encoding (shifted right)      │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Decoder Layer (×N)                      │     │
│  │  ┌─────────────────┐    ┌────────────┐ ┌────────┐  │     │
│  │  │  Masked MHA     │    │  Cross     │ │  FFN   │  │     │
│  │  │  (causal)       │    │  Attention │ │        │  │     │
│  │  │  Add & Norm     │    │  Add & Norm│ │Add&Norm│  │     │
│  │  └─────────────────┘    └────────────┘ └────────┘  │     │
│  └─────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│                      Linear + Softmax                         │
│                          ↓                                   │
│                        Output                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Encoder Layer 结构

每个 Encoder Layer 包含两个子层：

1. **Multi-Head Self-Attention**
2. **Feed-Forward Network (FFN)**

使用 **残差连接** (Residual Connection) 和 **Layer Normalization**：

$$
Output = \text{LayerNorm}(X + \text{SubLayer}(X))
$$

### 1.3 Decoder Layer 结构

每个 Decoder Layer 包含三个子层：

1. **Masked Multi-Head Self-Attention**（因果注意力，防止看到未来）
2. **Multi-Head Cross-Attention**（与 Encoder 输出交互）
3. **Feed-Forward Network**

---

## 2. Self-Attention 机制

### 2.1 本质

Self-Attention 的本质是：**对序列中的每个位置，计算它应该关注序列中其他所有位置的权重，然后加权求和得到新的表示**。

### 2.2 数学推导

设输入序列为 $\mathbf{X} = (x_1, x_2, ..., x_n)$, 每个 $x_i \in \mathbb{R}^{d_{model}}$. 

**Step 1: 线性变换得到 Q, K, V**

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V
$$

其中 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_k}$ 为可学习参数。

**Step 2: 计算注意力分数 (Attention Scores)**

$$
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}
$$

这里 $\mathbf{S} \in \mathbb{R}^{n \times n}$, $S_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的注意力权重（未归一化）。

**Step 3: Softmax 归一化**

$$
\mathbf{A} = \text{softmax}(\mathbf{S})
$$

其中 $\text{softmax}(\mathbf{S})_{ij} = \frac{e^{S_{ij}}}{\sum_{k=1}^{n} e^{S_{ik}}}$

**Step 4: 加权求和**

$$
\mathbf{O} = \mathbf{A} \mathbf{V}
$$

### 2.3 完整公式

$$
\boxed{\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}}
$$

### 2.4 为什么要除以 $\sqrt{d_k}$ ？

**原因：防止点积值过大导致 softmax 梯度消失**

设 $\mathbf{q}$ 和 $\mathbf{k}$ 是独立随机变量，均值为 0，方差为 1，则：

$$
\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{1}{\sqrt{d_k}}^2 \cdot \text{Var}(\mathbf{q} \cdot \mathbf{k}) = \frac{1}{\sqrt{d_k}}^2 \cdot d_k = 1
$$

但 $\mathbf{q} \cdot \mathbf{k}$ 的方差会随着 $d_k$ 增大而增大，导致 softmax 输入值过大，梯度趋近于 0。

除以 $\sqrt{d_k}$ 可以**保持点积结果的方差为 1**，确保训练稳定。

---

## 3. Multi-Head Attention

### 3.1 为什么需要多头注意力？

1. **多子空间学习**：不同的头可以关注不同的语义子空间
2. **增强表达能力**：单头注意力可能过度关注自身位置（身份认证问题）
3. **稳定训练**：多头提供更好的梯度流

### 3.2 数学公式

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \mathbf{W}^O
$$

其中每个 head 定义为：

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
$$

### 3.3 参数量分析

- 每个头的维度： $d_k = d_{model} / h$
- 参数量： $h \times (d_{model} \times d_k + d_k \times d_k + d_k \times d_{model}) = 4 \times d_{model}^2 + 2 \times d_{model} \times d_k \times h$
- 实际实现中，通常 $d_k = d_{model} / h$ ，使总参数量与单头注意力相当

### 3.4 实际代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projection and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(context)
```

---

## 4. 位置编码 (Positional Encoding)

### 4.1 为什么需要位置编码？

Attention 机制本身是**位置无关**的——对输入序列的每个位置一视同仁。Transformer 使用位置编码为序列注入位置信息。

### 4.2 Sinusoidal 位置编码

原始 Transformer 使用的位置编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中 $pos$ 是位置， $i$ 是维度索引。

### 4.3 性质

1. **每个位置有唯一编码**：不同位置的编码不同
2. **相对位置可表示**： $\sin(\alpha + \beta)$ 和 $\cos(\alpha + \beta)$ 可以线性组合表示相对位置
3. **可泛化到更长序列**：公式不依赖序列长度

### 4.4 其他位置编码

| 类型 | 描述 | 优缺点 |
|------|------|--------|
| **Sinusoidal** | 原始，正弦/余弦函数 | 可泛化，但表达能力有限 |
| **Learned Absolute PE** | 可学习的位置嵌入 | 需要更多参数，可能过拟合 |
| **RoPE (Rotary Position Embedding)** | 旋转式，LLaMA 使用 | 高效，支持相对位置 |
| **ALiBi** | 线性偏置，无需显式编码 | 适合长序列 |
| **K pos** | 注意力键的位置编码 | |

---

## 5. Layer Normalization

### 5.1 计算公式

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中：
- $\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$ （均值）
- $\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2$ （方差）
- $\gamma, \beta$ 为可学习参数

### 5.2 LayerNorm vs BatchNorm

| 特性 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| 归一化维度 | 单个样本的特征 | 批次维度 |
| 序列建模 | ✅ 适合变长序列 | ❌ 需要固定 batch |
| RNN | ✅ 常用 | 一般不用 |
| Transformer | ✅ 标准 | ❌ 不适合 |

**为什么 Transformer 用 LayerNorm？**

1. **序列长度可变**：不同样本的序列长度可能不同
2. **上下文依赖**：归一化应在单个样本内进行
3. **训练稳定性**：LayerNorm 提供更稳定的梯度

### 5.3 Pre-LN vs Post-LN

- **Post-LN**： $\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$ （原始论文）
- **Pre-LN**： $\text{Output} = x + \text{SubLayer}(\text{LayerNorm}(x))$

Pre-LN 在训练稳定性上更好，是现在的主流选择。

---

## 6. 前馈神经网络 (FFN)

### 6.1 结构

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

通常是两层全连接网络，带有 ReLU 激活。

### 6.2 参数量

对于隐藏维度 $d_{ff}$ ：

$$
\text{参数量} = 2 \times d_{model} \times d_{ff} + d_{ff} + d_{model}
$$

通常 $d_{ff} = 4 \times d_{model}$ 。

### 6.3 FFN 的作用

1. **非线性变换**：引入非线性，增加模型表达能力
2. **特征提取**：在注意力层后进一步变换表示
3. **计算占比**：FFN 约占 Transformer 1/3 的参数量

---

## 7. Encoder-Decoder vs Decoder-Only

### 7.1 架构对比

| 特性 | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| 代表模型 | T5, BART | GPT, LLaMA, PaLM |
| 注意力 | Cross-attention | 无 |
| 生成方式 | 非自回归可选 | 自回归 |
| 双向建模 | ✅ | ❌ |
| 条件生成 | 原生支持 | 需模板 |

### 7.2 为什么现在流行 Decoder-Only？

1. **统一范式**：预训练+微调都使用自回归
2. **简洁高效**：无需额外的 Cross-Attention
3. **涌现能力**：足够大的 Decoder-Only 模型表现出涌现能力
4. **工程实现**：推理优化更简单（KV Cache）

### 7.3 Decoder-Only 的注意力类型

| 类型 | 说明 |
|------|------|
| **Full Attention** | 全部 token 互相注意（用于训练） |
| **Causal Attention** | 只看前面的 token（自回归生成） |
| **Sliding Window** | 只看局部窗口（如 Mistral） |
| **Sparse Attention** | 稀疏模式（如 Longformer） |

---

## 8. 高频面试题

### Q1: Transformer 的基本原理是什么？

**参考答案**：

Transformer 是一种完全基于注意力机制的序列到序列模型，主要由 Encoder 和 Decoder 组成：

- **Encoder**：将输入序列 $(x_1, ..., x_n)$ 编码为上下文相关的表示 $(h_1, ..., h_n)$
- **Decoder**：基于 Encoder 的输出和已生成的部分，自回归地生成目标序列

核心是 **Multi-Head Self-Attention**，让序列中任意两个位置可以直接交互，摆脱了 RNN 的顺序依赖。

---

### Q2: 为什么要用多头注意力？ 一个头不够吗？

**参考答案**：

多头注意力相比单头有以下优势：

1. **多子空间表示**：不同头可以学习不同的语义关系（如语法、指代、语义等）
2. **增强鲁棒性**：一个头学到的模式可能被其他头补充
3. **稳定训练**：多头提供更多梯度路径

从参数效率角度，如果单头使用完整维度 $d_{model}$, 则参数量为 $O(d_{model}^2)$ ；而多头（ $h$ 个头）将维度分解为 $d_k = d_{model}/h$, 总参数量相当，但表达能力强得多。

---

### Q3: Attention 为什么要 scaled（除以 $\sqrt{d_k}$ ）？

**参考答案**：

是为了**数值稳定性**。

假设 $q$ 和 $k$ 的每个维度是独立随机变量，均值为 0，方差为 1，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$. 

当 $d_k$ 较大时，点积的方差会很大，导致 softmax 输入值偏向极大或极小，梯度接近于 0，训练困难。

除以 $\sqrt{d_k}$ 后，方差被归一化回 1，保证 softmax 有合适的梯度。

---

### Q4: Transformer 为什么用 LayerNorm 而不是 BatchNorm？

**参考答案**：

LayerNorm 和 BatchNorm 的核心区别在于归一化的维度：

- **BatchNorm**：对 batch 维度归一化，跨样本
- **LayerNorm**：对特征维度归一化，单样本内

Transformer 选择 LayerNorm 的原因：

1. **序列长度可变**：不同样本的序列长度可能不同
2. **样本独立**：NLP 任务中每个样本独立，不应跨样本归一化
3. **训练稳定性**：LayerNorm 提供更稳定的梯度

---

### Q5: 位置编码有哪些类型？ RoPE 是什么？

**参考答案**：

**常见位置编码类型**：

| 类型 | 特点 |
|------|------|
| Sinusoidal | 原始，可泛化到更长序列 |
| Learned | 可学习，可能过拟合 |
| RoPE | 旋转式，LLaMA 采用 |
| ALiBi | 线性偏置，长序列友好 |

**RoPE（Rotary Position Embedding）**：

RoPE 将位置信息编码为旋转矩阵，通过**在 Query 和 Key 上应用旋转操作**来实现相对位置编码：

$$
\mathbf{q}_m = \mathbf{R}_m \mathbf{q}, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{k}
$$

$$
\text{Attention}(\mathbf{q}_m, \mathbf{k}_n, \mathbf{v}_n) = \text{Attention}(\mathbf{R}_m \mathbf{q}, \mathbf{v}_n) \mathbf{R}_m \mathbf{R}_n^T
$$

RoPE 的优势：
- **高效**：无需额外参数
- **支持长序列**：相对位置编码
- **兼容性好**：可与 Flash Attention 结合

---

### Q6: Transformer 的计算复杂度是多少？

**参考答案**：

对于序列长度 $n$, 模型维度 $d$ ：

| 操作 | 复杂度 |
|------|--------|
| Self-Attention | $O(n^2 \cdot d)$ |
| FFN | $O(n \cdot d^2)$ |
| 参数量（每层） | $O(d^2)$ |

当 $n$ 较大时，Attention 的 $O(n^2)$ 成为瓶颈，因此有各种稀疏注意力变体（如 Longformer、BigBird）。

---

### Q7: 什么是 Masked Attention？ 怎么做？

**参考答案**：

Masked Attention（因果注意力）确保Decoder只能看到当前及之前的位置，不能"看到未来"。

**实现方式**：将注意力分数中对应"未来"位置的值设为 $-\infty$ ：

```python
seq_len = q.size(-2)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, -1e9)  # 上三角为 True 的位置填充 -inf
```

这样 softmax 后，"未来"位置的注意力权重为 0。

---

### Q8: Transformer 如何处理变长序列？

**参考答案**：

1. **Padding + Attention Mask**：对短序列补 0，用 mask 告诉模型哪些是 padding
2. **动态序列长度**：Attention 操作本身支持任意长度（通过 mask 控制）
3. **Bucketization**：将相近长度的序列 padding 到相同长度，减少无效计算

---

### Q9: Encoder 和 Decoder 的区别？

**参考答案**：

| 特性 | Encoder | Decoder |
|------|---------|---------|
| 注意力 | Bidirectional (可见全部) | Unidirectional (只看过去) |
| Cross-Attention | 无 | 有（与 Encoder 输出） |
| 用途 | 理解任务（分类、NER） | 生成任务（翻译、对话） |
| 典型应用 | BERT | GPT |

---

### Q10: Dropout 在 Transformer 中如何设置？

**参考答案**：

典型设置：
- **Attention Dropout**： $p=0.1$ ，对注意力权重 dropout
- **FFN Dropout**： $p=0.1$ ，对 FFN 输出 dropout
- **Embeddings Dropout**： $p=0.1$ ，对 embeddings dropout

注意：**推理时关闭 Dropout**，所有参数参与计算。


---

## 📝 面试高频题 & 参考答案

### Q1: Transformer 为何使用多头注意力，而非单头？

**参考答案：**

1. **增加表达能力**：不同的头可以关注不同的语义子空间，捕捉不同类型的关系（语法、语义、位置等）
2. **提高泛化能力**：如果只用一个头，参数矩阵是单一的；多头使整体参数矩阵形成多个子空间，增强模型对多方面信息的学习能力
3. **计算量基本不变**：多头只是将 $d_{model}$ 拆分为 $h$ 个 $d_k = d_{model}/h$ ，总计算量与单头接近，但表达能力大幅提升
4. **防止过度关注自身**：单头注意力在编码当前位置时容易过度聚焦自身，多头提供互补的注意力模式

> "类比 CNN 中的多通道滤波器——每个通道提取不同特征，多头注意力同理。"

---

### Q2: Attention 为什么要 Scaled（除以 $\sqrt{d_k}$ ）？

**参考答案：**

假设 $Q$ 和 $K$ 的各元素均值为 0、方差为 1，则 $QK^T$ 的每个元素是 $d_k$ 个独立随机变量的点积和，均值为 0，**方差为 $d_k$**。

- 当 $d_k$ 较大时，点积值幅度会很大，进入 softmax 的饱和区（梯度接近 0），导致训练困难
- 除以 $\sqrt{d_k}$ 后， $QK^T/\sqrt{d_k}$ 的方差恢复为 1，softmax 输入分布合适，梯度稳定

**数学推导：**

$$
	ext{Var}\left(rac{QK^T}{\sqrt{d_k}}
ight) = rac{1}{d_k} 	ext{Var}(QK^T) = rac{1}{d_k} \cdot d_k = 1
$$

---

### Q3: 为什么 Q 和 K 使用不同的权重矩阵，不能共用一个吗？

**参考答案：**

不能，原因有两点：

1. **解决输入/输出序列长度不一致问题**：Query 来自解码器（可变长度目标序列），Key 来自编码器（固定长度源序列），投影到不同空间才能独立处理不同序列长度
2. **增强泛化能力**：如果 $Q$ 和 $K$ 用同一权重矩阵（即 $K=Q$ ），则 $QK^T$ 是对称矩阵，attention score 矩阵具有对称性限制，模型表达能力大幅下降，无法学到非对称的注意力模式

---

### Q4: Transformer 计算 Attention 为何用点积而非加法？

**参考答案：**

- **表达能力强**：点积结果是有方向性的相似度度量， $W_Q$ 和 $W_K$ 将 $Q$ 和 $K$ 投影到不同空间，增加了非线性表达能力；而加法只是标量叠加，信息量少
- **计算效率高**：点积可以通过矩阵乘法 $QK^T$ 一次完成 $O(n^2 \cdot d)$ 复杂度；加法注意力（如Bahdanau）需要额外非线性层，计算更慢
- **两者效果相近时**：点积注意力在实践中表现更好，且更易于硬件加速（GPU/TPU 的矩阵乘法优化成熟）

---

### Q5: 为什么 Transformer 选择 LayerNorm 而不是 BatchNorm？

**参考答案：**

| 对比维度 | BatchNorm | LayerNorm |
|---------|-----------|-----------|
| **归一化维度** | 在 batch 维度上归一化 | 在特征维度上归一化 |
| **NLP 场景适配** | NLP 序列长度往往不一致，batch 统计不稳定 | 每个样本独立归一化，不依赖 batch |
| **序列生成任务** | 推理时需保存 running stats，复杂 | 推理/训练行为一致 |
| **RNN/LSTM** | 效果差 | 效果好 |

Transformer 中每个样本序列长度可能不同，且每个时刻需要基于自身统计归一化，所以 LayerNorm 更合适。

---

### Q6: 绝对位置编码和相对位置编码的区别？各有什么优缺点？

**参考答案：**

| 类型 | 代表方法 | 优点 | 缺点 |
|------|---------|------|------|
| **绝对位置编码** | 正弦/余弦编码、可学习 | 实现简单；外推性尚可（正弦） | 长度泛化差；每个位置独立建模 |
| **相对位置编码** | RoPE、ALiBi、T5 | 能显式建模 token 之间的距离；长度外推性好 | 实现复杂；部分方法需修改 Attention 逻辑 |

**RoPE（旋转位置编码）** 的核心思想是将位置信息旋转性地注入 Q 和 K，而不是直接加在 embedding 上，使得 $	ext{Attention}(Q_m, K_n) = 	ext{Attention}(\hat{Q}_m, \hat{K}_n)$ 自然包含相对位置 $m-n$ 。

---

### Q7: Decoder 中 Masked Multi-Head Attention 的作用是什么？

**参考答案：**

两种 Mask：
1. **Padding Mask**：将 padding 位置设为 $-\infty$ ，使其在 softmax 后权重为 0，避免无意义位置干扰
2. **Look-Ahead Mask（因果 Mask）**：将 $i$ 位置之后的位置设为 $-\infty$ ，保证解码器在预测第 $i$ 个 token 时只能看到之前（ $1 \sim i-1$ ）的信息，防止信息泄漏

**面试追问**：为什么不能像 Encoder 一样并行计算 Decoder？
> 因为解码器是自回归的，必须逐个 token 生成，当前 token 的生成依赖已生成的 token，所以需要 mask。

---

### Q8: Transformer 的残差连接有什么用？

**参考答案：**

1. **缓解梯度消失**：残差提供了一条梯度直接回传的路径，使得深层网络（104 层以上的 GPT）的训练成为可能
2. **稳定训练**：每层的输入是 $x + 	ext{SubLayer}(x)$ ，如果子层输出很小，恒等映射仍然保留；子层输出很大，就学一个新的映射
3. **特征复用**：底层特征可以无损传递到高层，有助于层级化表示的学习

---

### Q9: Transformer 训练时学习率如何设定？

**参考答案：**

使用 **Warmup + Cosine Decay** 策略：

$$
lr(t) = egin{cases}
lr_{	ext{max}} \cdot rac{t}{warmup\_steps} & t < warmup\_steps \
lr_{	ext{max}} \cdot rac{1}{2}\left(1 + \cos\left(rac{\pi (t - warmup\_steps)}{total\_steps - warmup\_steps}
ight)
ight) & t \ge warmup\_steps
\end{cases}
$$

- **Warmup 阶段**（通常 2000~10000 步）：学习率从很小逐步上升，防止早期因参数随机初始化导致的梯度爆炸
- **Cosine 衰减**：平滑下降，比 Step Decay 更稳定

---

### Q10: 什么是 WordPiece / BPE 分词？LLM 为何常用？

**参考答案：**

- **BPE（Byte Pair Encoding）**：从字符级别开始，贪心合并最高频的字符对，构建子词词表。GPT 系列使用
- **WordPiece**：基于语言学动机，合并那些能最大化训练数据似然的字符对。BERT 使用

**LLM 偏好子词分词的原因：**
1. **解决 OOV 问题**：稀有词/未登录词可被拆分为已知子词
2. **控制词表规模**：BPE/WP 的词表规模适中（通常 3~5 万），比词级词表小，比字符级词表语义更丰富
3. **中文化支持好**：中文可按字分或子词分，避免词库庞大

---

> 📚 **参考来源**：[CSDN - 史上最全Transformer面试题](https://blog.csdn.net/DeepLearning_/article/details/133986714)、[阿里云开发者社区 - Transformer面试题总结101道](https://developer.aliyun.com/article/904609)、[腾讯云 - Transformer灵魂22问](https://cloud.tencent.com/developer/article/2654837)
