# 大模型 & Agent 面试准备资料库

> 本仓库系统整理了大模型（LLM）和 AI Agent 岗位面试的核心知识点，涵盖 Transformer 架构、Attention 机制、训练微调、Agent 开发、RAG、推理优化等模块。每个模块均包含数学公式推导、面试高频问题及参考答案。

---

## 📚 目录结构

| 模块 | 主题 | 重要程度 |
|------|------|---------|
| [01-transformer-attention](./01-transformer-attention/) | Transformer 与 Attention 机制 | ⭐⭐⭐⭐⭐ |
| [02-llm-fundamentals](./02-llm-fundamentals/) | LLM 基础概念与架构 | ⭐⭐⭐⭐⭐ |
| [03-training-finetuning](./03-training-finetuning/) | 训练、微调、RLHF/DPO | ⭐⭐⭐⭐⭐ |
| [04-agent](./04-agent/) | Agent 智能体原理与架构 | ⭐⭐⭐⭐⭐ |
| [05-rag](./05-rag/) | RAG 检索增强生成 | ⭐⭐⭐⭐⭐ |
| [06-optimization](./06-optimization/) | 推理优化、量化、并行 | ⭐⭐⭐⭐ |
| [07-moe](./07-moe/) | MoE 混合专家架构 | ⭐⭐⭐ |
| [08-prompt-engineering](./08-prompt-engineering/) | Prompt 工程核心技巧 | ⭐⭐⭐⭐ |
| [09-context-engineering](./09-context-engineering/) | 上下文管理、RAG、长上下文 | ⭐⭐⭐⭐ |
| [10-evaluation-harness](./10-evaluation-harness/) | 评测框架、Benchmark 构建 | ⭐⭐⭐ |
| [11-skill-tool-use](./11-skill-tool-use/) | 工具调用、Function Calling | ⭐⭐⭐⭐ |

---

## 🔑 核心面试高频考点

### 必须掌握的数学公式

**1. Scaled Dot-Product Attention**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**2. Layer Normalization**

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中 $\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$, $\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2$

**3. Positional Encoding (Sinusoidal)**

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**4. RLHF / PPO 目标函数**

$$
L_{\text{PPO}} = \mathbb{E}_{(s,a)\sim\pi_{\theta_{\text{old}}}} \left[ \min\left( r(\theta) \hat{A}_{\text{GAE}}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{\text{GAE}} \right) \right]
$$

其中 $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$

**5. DPO 损失函数**

$$
L_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r(x, y_w) - r(x, y_l) \right) \right]
$$

其中 $r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

---

## 📖 使用说明

1. **按模块学习**：建议从 01-05 顺序学习，最后看 06-07
2. **理解公式**：每个模块都附有详细的数学推导和物理意义
3. **准备面试**：每个模块附有高频面试题，可用于自测
4. **项目落地**：部分模块包含代码示例和工程实践

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

*Last updated: 2026-04-22*
