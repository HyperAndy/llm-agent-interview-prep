# Context Engineering

Context Engineering（上下文工程）是指对 LLM 的输入上下文进行系统性设计、压缩、检索、增强和迭代优化的工程实践。是构建高效 RAG、Agent 和多轮对话系统的核心能力。

---

## 1. 上下文的核心组成

```
┌─────────────────────────────────────────────────────┐
│                   输入上下文                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────┐ │
│  │ System    │  │ User      │  │ Tool Results /    │ │
│  │ Prompt    │  │ Query     │  │ World Knowledge   │ │
│  └───────────┘  └───────────┘  └───────────────────┘ │
│       ↑              ↑                  ↑            │
│   角色定义       用户意图理解         外部知识增强     │
└─────────────────────────────────────────────────────┘
```

### 1.1 System Prompt（系统级上下文）

定义模型行为模式、角色、知识边界和输出格式。

```python
SYSTEM_PROMPT = """你是一位专业的股权投资分析师，专注 A 股市场。
能力范围：
- 财报关键指标提取与横向对比
- 行业景气度判断
- 估值方法：DCF、PE、PB 比较

约束：
- 数据引用必须标注来源
- 涉及财务预测必须说明假设
- 避免荐股，仅提供分析框架

输出格式：
## 分析结论
[核心观点，3 条以内]

## 数据支撑
[关键数据表格]

## 风险提示
[主要风险因素]
"""
```

### 1.2 Conversation History（对话历史上下文）

多轮对话中保留关键历史信息：

```python
def build_conversation_history(messages, max_turns=10):
    """保留最近 N 轮对话，避免上下文超出限制"""
    return messages[-max_turns * 2:]  # 每轮包含 user + assistant

# 关键策略：摘要压缩（Summarize + Compress）
def compress_history(messages):
    """对长对话历史进行摘要，保留关键实体和结论"""
    summary_prompt = """将以下对话摘要为 3-5 句核心内容：
    保留：关键决策、悬而未决的问题、用户偏好、重要事实
    [对话内容]"""
    summary = model.ask(summary_prompt)
    return [summary]
```

---

## 2. 上下文长度管理

### 2.1 Token 预算控制

```python
def count_tokens(text, model="gpt-4"):
    """估算 token 数量（粗略：中文 1 token ≈ 1.5 字符）"""
    return int(len(text) / 1.5)

def fit_in_budget(context_items, budget=128000):
    """贪心策略：将上下文项目按重要性排序，塞满预算"""
    total = 0
    selected = []
    for item in sorted(context_items, key=lambda x: x.priority, reverse=True):
        if total + item.tokens <= budget:
            selected.append(item)
            total += item.tokens
    return selected
```

### 2.2 上下文截断策略

| 策略 | 适用场景 | 风险 |
|------|----------|------|
| **Fixed Window** | 固定保留开头+结尾 | 中间重要信息丢失 |
| **Sliding Window** | 流式对话 | 关键信息可能刚好被滑掉 |
| **Priority-Based** | 按重要性选择 | 需要定义优先级 |
| **Recency-Biased** | 优先保留最近 | 早期关键事实丢失 |
| **Semantic Chunking** | 按语义分段选择 | 语义边界判断难 |

---

## 3. RAG 上下文增强

### 3.1 检索增强生成（RAG）核心流程

```
用户问题 → 检索（Query Rewrite）→ 文档块（Top-K）→ 上下文组装 → LLM 生成
           ↑                                      ↓
        意图澄清                           引用标注 + 置信度
```

### 3.2 Query Rewrite（查询改写）

将用户原始问题改写为更适合检索的形式：

```python
# 隐式问题显式化
original: "它的副作用是什么？"  # "它" 指代不明
rewritten: "某降压药的副作用是什么？"

# 口语转书面
original: "我想知道这个股票能不能买"
rewritten: "某股票的投资价值与风险分析"

# 拆分为多个子查询
original: "Transformer 的原理和 GPT 的区别是什么？"
rewritten: ["Transformer 注意力机制原理", "GPT 模型架构与训练"]
```

### 3.3 检索结果排序与重排（Rerank）

```python
# 第一阶段：向量检索（高效、语义匹配）
initial_results = vector_db.search(query, top_k=50)

# 第二阶段：Cross-Encoder 重排（精准、考虑查询-文档相关性）
reranked = cross_encoder.rerank(query, initial_results, top_k=10)

# Rerank 模型示例
# bge-reranker-large、Cohere rerank
```

### 3.4 上下文组装策略

**Naive RAG**：直接将 Top-K 检索结果拼接。

**Parent Document Retriever**：
```python
# 小块用于精确匹配，匹配到后附加完整父文档
small_chunks = vector_db.search(query, top_k=20)
parent_ids = set(c.parent_id for c in small_chunks)
full_docs = [db.get(id) for id in parent_ids]
```

**Sentence Window Retrieval**：
```python
# 匹配到句子后，扩展周围窗口的上下文
window = 3  # 前后各 3 句
expanded = sentence.expand_window(matched_sentence, window)
```

---

## 4. 上下文压缩

### 4.1 知识库压缩

```python
# LLMLingua：利用小型语言模型压缩 Prompt
from llmlingua import PromptCompressor

compressor = PromptCompressor(model="YeungNLP/llmlingua-2-bert-base-multilingual-cased-meetingbank")
compressed = compressor.compress_prompt(
    prompt,
    rate=0.5,  # 压缩到 50%
    force_context_tokens=500
)
```

### 4.2 文档块压缩

```python
# 递归字符分割
def split_with_overlap(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 语义分割（按段落/主题）
# 使用模型判断句子边界
```

---

## 5. 上下文窗口扩展

### 5.1 Long Context 模型

| 模型 | Context 长度 | 适用场景 |
|------|-------------|----------|
| Claude 3/4 | 200K | 长文档分析、多轮对话 |
| Gemini 1.5 | 1M/2M | 超长上下文任务 |
| GPT-4 Turbo | 128K | 中长文档处理 |
| Kimi (月之暗面) | 200K+ | 中文超长上下文 |
| Yi (零一) | 200K | 中文长上下文 |

### 5.2 Long Context 技术

**稀疏注意力（Sparse Attention）**：只关注局部窗口和重要位置。

**滑动窗口注意力（Sliding Window）**：
```python
# 每个 token 只attend到前后 window_size 的范围
attention_mask = torch.tril(torch.ones(window_size, window_size))
```

**分段处理（Chunked Processing）**：
```python
def process_long_doc(doc, chunk_size=8000, overlap=200):
    """分段处理，最后汇总"""
    chunks = split_with_overlap(doc, chunk_size, overlap)
    summaries = [summarize(chunk) for chunk in chunks]
    return synthesize(summaries)
```

**Longformer/LongChat**：专门针对长上下文的 Transformer 变体。

---

## 6. 上下文一致性

### 6.1 多文档一致性

当多个检索文档对同一事实描述不一致时：

```python
def resolve_conflicts(docs):
    """冲突解决策略"""
    # 1. 信任优先级：官方文件 > 权威媒体 > 自媒体
    # 2. 多数一致性：多文档共同描述更可信
    # 3. 时效优先：最新日期优先
    # 4. 明确标注：无法判断时标注"存在不同说法"
    
    return merge_with_conflict_flag(docs)
```

### 6.2 上下文幻觉检测

```python
# 检查 LLM 生成是否与检索上下文一致
def verify_grounding(response, context):
    """检测生成内容是否被上下文支持"""
    claims = extract_claims(response)
    supported = []
    unsupported = []
    for claim in claims:
        if any(claim_in_text(claim, doc) for doc in context):
            supported.append(claim)
        else:
            unsupported.append(claim)
    return supported, unsupported
```

---

## 7. 动态上下文

### 7.1 自适应上下文

```python
class AdaptiveContext:
    def __init__(self, budget=128000):
        self.budget = budget
    
    def build(self, task, query):
        # 简单任务 → 少上下文
        if task.difficulty == "simple":
            return self._lightweight(query)
        # 复杂任务 → 多上下文 + 推理链
        elif task.difficulty == "complex":
            return self._heavyweight(query)
```

### 7.2 上下文缓存（KV Cache）

```python
# 已在 attention 层缓存的 KV 不重新计算
# vLLM/Flash Attention 等框架自动支持

# 主动缓存：固定前缀（如 System Prompt）只计算一次
cached_prefix = cache.compute(system_prompt)
```

---

## 8. 参考资料

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 长上下文检索问题
- [RAG vs. Fine-tuning vs. Prompt Engineering](https://www.prompteng.ai/research/rag-vs-ft) — 选择策略
- [LLMLingua: Compressing Prompt for Intelligence](https://arxiv.org/abs/2310.05736)
