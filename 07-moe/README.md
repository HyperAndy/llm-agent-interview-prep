# Module 07: MoE 混合专家架构

## 📋 目录

1. [MoE 概述](#1-moe-概述)
2. [MoE 数学原理](#2-moe-数学原理)
3. [负载均衡](#3-负载均衡)
4. [主流 MoE 模型](#4-主流-moe-模型)
5. [高频面试题](#5-高频面试题)

---

## 1. MoE 概述

### 1.1 什么是 MoE？

**MoE (Mixture of Experts)**：混合专家架构，通过稀疏激活机制，在不增加计算量的情况下大幅扩展模型参数。

### 1.2 MoE vs 稠密模型

| 维度 | 稠密模型 (Dense) | MoE |
|------|-----------------|-----|
| 参数量 | $N$ | $N \times n_{\text{experts}}$ |
| 每次前向计算量 | 固定 | 稀疏（只激活 Top-K） |
| 显存 | 高 | 中等（路由开销） |
| 计算效率 | 100% | 取决于稀疏度 |

---

## 2. MoE 数学原理

### 2.1 稀疏门控 (Sparse Gating)

对于输入 $x$ ，MoE 层输出为：

$$
y = \sum_{i=1}^{E} G(x)_i \cdot E_i(x)
$$

其中：
- $E_i$ 是第 $i$ 个 Expert（通常是 FFN）
- $G(x)_i$ 是路由权重

**稀疏激活**： $G(x)$ 只有 Top-K 个非零值

### 2.2 Router (Gating Network)

$$
G(x) = \text{Softmax}(\text{TopK}(W_g x))
$$

**公式分解**：
1. 线性变换： $z = W_g x \in \mathbb{R}^E$
2. TopK 选择：保留最大的 $K$ 个值，其他置为 $-\infty$
3. Softmax 归一化

---

## 3. 负载均衡

### 3.1 负载均衡问题

稀疏路由可能导致**负载不均衡**：
- 某些 Expert 被频繁选中，过载
- 其他 Expert 几乎不被选中，浪费

### 3.2 辅助损失 (Auxiliary Loss)

**MLP 辅助损失**（Mixtral 采用）：

$$
\mathcal{L}_{\text{aux}} = \frac{1}{E} \sum_{e=1}^{E} f_e \cdot p_e
$$

其中：
- $f_e$ ：第 $e$ 个 Expert 被选中的频率
- $p_e$ ：路由对 Expert $e$ 的平均概率

### 3.3 偏置项调整

**Expert 偏置项**（Switch Transformer 采用）：

$$
G(x)_i = \text{Softmax}(\text{TopK}(W_g x + b_i))
$$

动态调整偏置项： $b_i \leftarrow b_i - \lambda \cdot (f_i - \text{target})$

---

## 4. 主流 MoE 模型

### 4.1 模型对比

| 模型 | 参数量 | 活跃参数 | Experts | Top-K | 特点 |
|------|--------|----------|---------|-------|------|
| **Switch Transformer** | 1.6T | 6B | 2048 | 1 | 极致稀疏 |
| **Mixtral 8x7B** | 46.7B | 12.9B | 8 | 2 | 平衡高效 |
| **DBRX** | 132B | 36B | 16 | 4 | GPT-4 级 |
| **DeepSeek-MoE** | 145B | 14B | 64 | 2 | 细粒度 |

### 4.2 Mixtral 8x7B 架构

- 8 个 Expert（每个是标准 FFN）
- Router 选择 Top-2 Expert
- 每次前向：激活 2/8 = 25% 的 FFN
- 与 46.7B Dense 模型效果相当，速度快 2x

**配置**：32 layers，Hidden size: 14336，Attention heads: 8，Expert FFN hidden: 43008

---

## 5. 高频面试题

### Q1: MoE 的原理是什么？

**参考答案**：

MoE (Mixture of Experts) 通过**稀疏激活**实现高效的大模型：

1. **多专家**：多个独立的 FFN（Experts）
2. **稀疏路由**：每个 token 只激活 Top-K 个 Expert
3. **加权求和**：根据路由权重组合 Expert 输出

$$
y = \sum_{i=1}^{E} G(x)_i \cdot E_i(x)
$$

相比 Dense 模型，MoE 用更少的计算量实现更大参数量的模型。

---

### Q2: MoE 相比 Dense 模型的优势？

**参考答案**：

| 优势 | 说明 |
|------|------|
| **参数效率** | 参数量大但计算量小 |
| **训练速度** | 稀疏激活，训练更快 |
| **推理效率** | 只需计算激活的 Expert |
| **表达能力** | 多个 Expert 可学习不同能力 |

---

### Q3: MoE 的负载均衡问题？

**参考答案**：

稀疏路由可能导致 Expert 利用不均：

**解决方案**：
1. **辅助损失**：加入负载均衡正则项
2. **Expert 偏置调整**：动态调整偏置项
3. **噪声路由**：加入随机性促进均衡

---

### Q4: MoE 的训练挑战？

**参考答案**：

| 挑战 | 描述 |
|------|------|
| **负载均衡** | Expert 利用不均 |
| **通信开销**（分布式） | Expert 在不同设备需通信 |
| **显存不均** | 不同设备 Expert 显存差异 |
| **训练稳定性** | 稀疏路由梯度不稳定 |
