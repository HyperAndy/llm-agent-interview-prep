# Module 02: LLM 大模型基础概念

## 📋 目录

1. [LLM 发展历程](#1-llm-发展历程)
2. [主流模型架构对比](#2-主流模型架构对比)
3. [分词技术 (Tokenization)](#3-分词技术-tokenization)
4. [语言建模目标](#4-语言建模目标)
5. [涌现能力 (Emergent Abilities)](#5-涌现能力-emergent-abilities)
6. [上下文窗口与长度外推](#6-上下文窗口与长度外推)
7. [高频面试题](#7-高频面试题)

---

## 1. LLM 发展历程

### 1.1 GPT 系列

| 模型 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-1 | 2018 | 117M | 开创性 |
| GPT-2 | 2019 | 1.5B | 更大、zero-shot |
| GPT-3 | 2020 | 175B | Few-shot, in-context learning |
| GPT-3.5 | 2022 | - | ChatGPT |
| GPT-4 | 2023 | - | 多模态、长上下文 |
| GPT-4o | 2024 | - | 原生多模态 |

### 1.2 BERT 系列

| 模型 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| BERT-Base | 2018 | 110M | 双向编码 |
| BERT-Large | 2018 | 340M | 更大 |
| RoBERTa | 2019 | 355M | 改进预训练 |
| ALBERT | 2020 | 12M (12层) | 参数共享 |

### 1.3 开源模型

| 模型 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| LLaMA | 2023 | 7B-65B | 高效开源 |
| LLaMA 2 | 2023 | 7B-70B | 开源可商用 |
| LLaMA 3 | 2024 | 8B-405B | 超过 GPT-3.5 |
| Mistral | 2023 | 7B | Sliding Window |
| Qwen | 2024 | 7B-72B | 阿里中文优化 |
| DeepSeek | 2024 | 7B-67B | 高性价比 |

---

## 2. 主流模型架构对比

### 2.1 BERT vs GPT vs T5

| 特性 | BERT | GPT | T5 |
|------|------|-----|-----|
| 架构 | Encoder-only | Decoder-only | Encoder-Decoder |
| 注意力 | Bidirectional | Unidirectional | Bidirectional (Enc) + Unidirectional (Dec) |
| 预训练目标 | MLM + NSP | CLM (Next Token) | Span Corruption |
| 擅长任务 | 理解（分类、NER） | 生成（对话、写作） | 条件生成（翻译、摘要） |
| 代表应用 | 文本分类 | ChatGPT | 文生文 |

### 2.2 架构选择的影响

**为什么 GPT 用 Decoder-only？**

1. **工程简洁**：预训练和推理统一为自回归
2. **涌现能力**：足够大的 Decoder-only 模型涌现出 ICM（上下文学习）
3. ** Scaling Law**：Decoder-only 在 Scaling 上表现更好
4. **RLHF 友好**：直接对应人类反馈强化学习

---

## 3. 分词技术 (Tokenization)

### 3.1 常见分词方法

| 方法 | 描述 | 优缺点 |
|------|------|--------|
| **WordPiece** | BERT 使用，基于频率的子词 | 词汇量可控 |
| **BPE** (Byte Pair Encoding) | 贪心合并高频字节对 | 简单高效 |
| **SentencePiece** | BPE 变体，支持多语言 | 无需预切分 |
| **Unigram** | LLaMA 使用，概率模型 | 更灵活 |

### 3.2 BPE 算法步骤

```
输入: 训练语料
输出: 词汇表

1. 初始化词汇表为所有字符
2. 统计相邻字符对 (bigram) 频率
3. 合并最高频的字符对，加入词汇表
4. 重复步骤 2-3，直到达到词汇表大小
```

### 3.3 分词对 LLM 的影响

- **词汇表大小**：通常 32k-200k tokens
- **中文字符**：中文需要更大的词汇表或特殊处理
- **压缩率**：英文约 0.25 tokens/字符，中文约 1.5-2 tokens/字

---

## 4. 语言建模目标

### 4.1 掩码语言建模 (MLM)

**BERT 使用**，随机遮盖 15% 的 tokens，预测被遮盖的部分：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\setminus i})
$$

### 4.2 因果语言建模 (CLM)

**GPT 使用**，基于前面的 tokens 预测下一个 token：

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

### 4.3 Span Corruption

**T5 使用**，随机遮盖连续的 span，用 `[MASK]` 替换：

$$
\mathcal{L} = -\sum_{i \in \text{spans}} \log P(x_i \mid x_{\setminus \text{spans}})
$$

### 4.4 对比

| 目标 | 模型 | 双向建模 | 生成能力 |
|------|------|---------|---------|
| MLM | BERT | ✅ | ❌ |
| CLM | GPT | ❌ | ✅ |
| Span Corruption | T5 | Encoder 双向 | ✅ |

---

## 5. 涌现能力 (Emergent Abilities)

### 5.1 定义

涌现能力是指模型在训练过程中**突然表现出的新能力**，这些能力在小模型中不存在或不可用。

### 5.2 典型涌现能力

| 能力 | 描述 | 涌现规模 |
|------|------|---------|
| **In-Context Learning** | 从提示中学习新任务 | ~100B |
| **Chain-of-Thought** | 推理时逐步思考 | ~100B |
| **Instruction Following** | 遵循指令 | ~10B (FLAN) |
| **Code Generation** | 生成代码 | ~10B |
| **Math Reasoning** | 数学推理 | ~100B |

### 5.3 涌现能力的原因

**有争议的话题**，主要有两种观点：

1. **定量涌现**：能力不是突然出现的，而是指标选择不当导致
2. **定性涌现**：大模型确实发展出小模型没有的新能力

**Scaling Hypothesis**：
$$
\text{能力}(N, D) = f\left(\frac{N}{N_{\text{threshold}}}, \frac{D}{D_{\text{threshold}}}\right)
$$

当模型规模 $N$ 和数据量 $D$ 超过阈值时，能力突然出现。

---

## 6. 上下文窗口与长度外推

### 6.1 上下文窗口

上下文窗口定义了 LLM 一次能处理的**最大 token 数量**：

| 模型 | 上下文窗口 |
|------|-----------|
| GPT-3 | 2,049 |
| GPT-4 | 128k |
| Claude 2 | 200k |
| LLaMA 3 | 128k |
| DeepSeek | 128k |

### 6.2 长度外推问题

标准 Transformer 的 RoPE/位置编码在**训练长度外**表现下降。

### 6.3 长度外推技术

**1. RoPE (Rotary Position Embedding)**

RoPE 将绝对位置编码为旋转矩阵：

对于位置 $m$ 的 query $\mathbf{q}_m$：
$$
\mathbf{q}_m = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m
$$

其中 $\mathbf{R}_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$

**2. Linearized Attention**

将 $\exp(QK^T)$ 线性化，使复杂度降为 $O(n)$. 

**3. Long Context Calibration**

在推理时使用更长上下文，但可能需要校准。

---

## 7. 高频面试题

### Q1: GPT 和 BERT 的区别是什么？

**参考答案**：

| 维度 | GPT | BERT |
|------|-----|------|
| 架构 | Decoder-only | Encoder-only |
| 注意力 | Causal (单向) | Bidirectional (双向) |
| 预训练目标 | 因果语言建模 (Next Token) | 掩码语言建模 (MLM) |
| 擅长任务 | 文本生成、对话 | 理解任务（分类、NER） |
| 生成方式 | 自回归 | 非自回归 |

GPT 单向注意力适合生成，BERT 双向注意力适合理解。

---

### Q2: 什么是 In-Context Learning？

**参考答案**：

In-Context Learning (ICL) 是指 LLM 在不更新参数的情况下，通过提示中的示例学习新任务。

**数学形式**：
$$
P(y \mid x, \text{demos}) = P(y \mid x, \{(x_i, y_i)\}_{i=1}^{k})
$$

其中 demos 是 $k$ 个示例 $(x_i, y_i)$. 

**关键发现**：ICM 能力随模型规模涌现（~100B 参数），小模型不具备。

---

### Q3: LLM 的幻觉问题是什么？如何缓解？

**参考答案**：

**幻觉 (Hallucination)**：LLM 生成的内容看似合理但实际错误或无根据。

**原因**：
1. 训练数据中的噪声
2. 知识过时
3. 推理过程中的一味"concise"倾向

**缓解方法**：

| 方法 | 描述 |
|------|------|
| RAG | 检索外部知识，约束生成 |
| 引用来源 | 让模型引用参考资料 |
| 微调 | 使用高质量对齐数据 |
| 提示工程 | Chain-of-Thought、Self-Check |
| Decoding 策略 | 降低 temperature，增加确定性 |

---

### Q4: 为什么现在大模型大多用 Decoder-only？

**参考答案**：

1. **涌现能力**：足够大的 Decoder-only 模型涌现出 ICM 和 CoT
2. **工程统一**：预训练和推理范式一致
3. **Scaling 友好**：Decoder-only 在 Scaling Law 上表现更好
4. **RLHF 对齐**：直接对应人类偏好学习

Encoder-Decoder 模型（如 T5）虽然理解能力强，但生成能力和涌现能力不如同等规模的 Decoder-only。

---

### Q5: 什么是 Chain-of-Thought？

**参考答案**：

Chain-of-Thought (CoT) 是一种推理策略，让模型**显式生成推理步骤**再给出答案。

**效果**：
- 简单问题：效果不明显
- 复杂推理（数学、逻辑）：显著提升准确率

**数学解释**：
$$
P(\text{answer} \mid \text{problem}) = \sum_{\text{chain}} P(\text{chain} \mid \text{problem}) \cdot P(\text{answer} \mid \text{chain})
$$

通过显式建模推理链，模型能更好地处理多步推理。

---

### Q6: LLM 的参数量怎么计算？

**参考答案**：

以 LLaMA-7B 为例：

| 组件 | 计算 |
|------|------|
| Embedding | $V \times d = 32000 \times 4096 \approx 131M$ |
| 每层 Attention | $4 \times d_{model}^2 = 4 \times 4096^2 \approx 67M$ |
| 每层 FFN | $2 \times d_{model} \times d_{ff} = 2 \times 4096 \times 11008 \approx 90M$ |
| 每层 LayerNorm | $4 \times d_{model} \approx 16K$ |
| 总计（32层） | $\approx 7B$ |

---

### Q7: 什么是 Prefix LM 和 Casual LM？

**参考答案**：

| 类型 | 注意力模式 | 应用 |
|------|-----------|------|
| **Causal LM** | 纯单向，只能看过去 | GPT 系列 |
| **Prefix LM** | 前缀双向，生成时单向 | T5、GLM |
| **Permutation LM** | 随机顺序 | XLNet |

Prefix LM 在 Prefix 部分使用双向注意力，在预测部分使用单向注意力。

---

### Q8: 如何评估 LLM 的性能？

**参考答案**：

**常用基准**：

| 基准 | 考察能力 |
|------|---------|
| MMLU | 多任务知识 |
| HumanEval | 代码生成 |
| GSM8K | 数学推理 |
| BIG-Bench | 综合能力 |
| HELM | 多维度 |

**评估维度**：

1. **准确性**：任务完成度
2. **效率**：推理速度、显存占用
3. **安全性**：有害输出、偏见
4. **有用性**：符合人类偏好
