# Module 06: 推理优化与部署

## 📋 目录

1. [推理优化概述](#1-推理优化概述)
2. [KV Cache 与 Beam Search](#2-kv-cache-与-beam-search)
3. [量化技术](#3-量化技术)
4. [注意力优化](#4-注意力优化)
5. [并行策略](#5-并行策略)
6. [推理框架](#6-推理框架)
7. [高频面试题](#7-高频面试题)

---

## 1. 推理优化概述

### 1.1 推理效率瓶颈

LLM 推理的两个阶段：

| 阶段 | 特点 | 瓶颈 |
|------|------|------|
| **Prefill** | 处理输入 prompt，计算 KV Cache | 计算密集 |
| **Decode** | 自回归生成 token | 访存密集 |

### 1.2 延迟与吞吐量

**延迟 (Latency)**：单个请求的响应时间

$$
\text{Latency} = T_{\text{prefill}} + N \times T_{\text{decode}}
$$

**吞吐量 (Throughput)**：

$$
\text{Throughput} = \frac{\text{Batch Size}}{\text{Total Time}}
$$

---

## 2. KV Cache 与 Beam Search

### 2.1 KV Cache 原理

自回归生成时，已计算的 Key 和 Value 可缓存复用：

**无 KV Cache**：每个 token 需 $O(n \cdot d)$ （重复计算）

**有 KV Cache**：
- Prefill： $O(n \cdot d)$
- Decode： $O(1 \cdot d)$ （复用）

### 2.2 KV Cache 显存占用

对于半精度 (bf16) 的 KV Cache：

$$
\text{KV Cache} \approx 2 \times n_{\text{layers}} \times s_{\text{max}} \times d_{\text{model}} \times 2 \text{ bytes/token}
$$

对于 7B 模型（seq_len=2048）：约 16GB！

### 2.3 Beam Search

在每一步保留 Top-B 候选路径，与 Greedy/Sampling 对比：

| 方法 | 特点 | 质量 | 速度 |
|------|------|------|------|
| Greedy | 始终选最高概率 | 中等 | 快 |
| Beam Search | 保留 Top-B 路径 | 高 | 中 |
| Sampling | 随机采样 | 可变 | 中 |

---

## 3. 量化技术

### 3.1 量化基本概念

**量化 (Quantization)**：将高精度（fp32/bf16）参数映射到低精度（int8/int4）表示。

### 3.2 量化方法分类

| 类型 | 描述 | 精度损失 |
|------|------|----------|
| **PTQ** (Post-Training) | 训练后量化 | 中等 |
| **QAT** (Quantization-Aware) | 量化感知训练 | 低 |
| **动态量化** | 推理时动态量化 | 中等 |
| **静态量化** | 提前量化参数 | 低 |

### 3.3 常见量化格式

| 格式 | 精度 | 显存节省 | 速度 |
|------|------|----------|------|
| FP16 | 16-bit | 1x | baseline |
| BF16 | 16-bit | 1x | ~1x |
| INT8 | 8-bit | 2x | 1.5-2x |
| INT4 | 4-bit | 4x | 2-4x |

### 3.4 GPTQ 量化

**GPTQ (Generative Post-Training Quantization)**：

核心思想：逐层量化，最小化重建误差。

$$
\arg\min_{\hat{W}} \|W - \hat{W}\|_F^2 \quad \text{s.t.} \hat{W} \in \mathbb{Z}^n
$$

### 3.5 GGUF/llama.cpp 量化

| 类型 | Bits | 特点 |
|------|------|------|
| Q4_K_M | 4-bit | 中等质量，内存效率平衡 |
| Q4_K_S | 4-bit | 更高质量，稍多内存 |
| Q5_K_M | 5-bit | 高质量 |
| Q8_0 | 8-bit | 接近 FP16 |

---

## 4. 注意力优化

### 4.1 Flash Attention

**核心思想**：将 Attention 计算分块，避免物化完整注意力矩阵。

**标准 Attention**：需要 $O(n^2)$ 显存存储 $S = QK^T$

**Flash Attention**：只需 $O(n)$ 显存

核心是「Tiling」+「Online Softmax」算法。

### 4.2 Page Attention

**vLLM 采用**，灵感来自操作系统虚拟内存分页。

**解决的问题**：KV Cache 显存碎片化

**核心思想**：
- 将 KV Cache 分成固定大小的块（Page）
- 动态分配，按需使用
- 显著提高 batch size 和吞吐量

---

## 5. 并行策略

### 5.1 各种并行方式

| 并行方式 | 切分维度 | 通信量 | 适用场景 |
|----------|----------|--------|----------|
| **Data Parallel** | 数据 | 低 | 多卡 |
| **Tensor Parallel** | 张量 | 高 | 单节点多卡 |
| **Pipeline Parallel** | 层 | 中 | 多节点 |
| **ZeRO** | 优化器状态 | 低 | 多节点 |

### 5.2 Continuous Batching

**传统 Batching**：等所有请求到达才一起处理

**Continuous Batching**：
- 新请求到达立即加入 batch
- 已完成的请求及时移出
- 显著提高 GPU 利用率

---

## 6. 推理框架

### 6.1 主流推理框架对比

| 框架 | 开发 | 特点 | 量化支持 |
|------|------|------|----------|
| **vLLM** | UC Berkeley | PagedAttention，高吞吐 | INT4/8 |
| **TensorRT-LLM** | NVIDIA | 高度优化，推理快 | INT8/FP8 |
| **llama.cpp** | Georgi Gerganov | 纯 CPU/GPU，移植性好 | INT4/8 |
| **Ollama** | 社区 | 本地运行，易用 | INT4/8 |

### 6.2 vLLM 核心优化

vLLM 的核心是 **PagedAttention**：
1. **分页管理 KV Cache**：物理块模拟逻辑 KV Cache
2. **动态分配**：按需分配块，避免预分配浪费
3. **共享块**：支持前缀复用（如多轮对话）
4. **Continuous Batching**：动态批处理，提高 GPU 利用率

---

## 7. 高频面试题

### Q1: LLM 推理的两个阶段是什么？

**参考答案**：

1. **Prefill 阶段**：处理输入 prompt，计算 KV Cache，计算密集
2. **Decode 阶段**：自回归生成 token，访存密集（读写 KV Cache）

---

### Q2: 什么是 KV Cache？

**参考答案**：

KV Cache 在自回归生成时缓存已计算的 Key-Value 注意力状态，避免重复计算历史 token 的 K/V，大幅降低 Decode 阶段的计算量。

---

### Q3: 量化是什么？ INT8 和 INT4 的区别？

**参考答案**：

量化是将高精度参数映射到低精度表示的技术：

| 精度 | 显存节省 | 速度提升 |
|------|----------|----------|
| INT8 | 2x | 1.5-2x |
| INT4 | 4x | 2-4x |

INT4 显存节省更多，但精度损失更大。

---

### Q4: Flash Attention 的原理？

**参考答案**：

Flash Attention 通过**分块计算**避免物化完整注意力矩阵。标准 Attention 需要 $O(n^2)$ 显存，Flash Attention 只需 $O(n)$. 核心是「Tiling」+「Online Softmax」算法。

---

### Q5: vLLM 的核心优化？

**参考答案**：

vLLM 的核心是 **PagedAttention**：将 KV Cache 逻辑上连续、物理上分块管理，动态分配 + 支持共享块 + Continuous Batching，显著提高吞吐量。
