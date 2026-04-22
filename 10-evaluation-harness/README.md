# Evaluation Harness

评测体系是 LLM 开发闭环的核心。本模块覆盖主流评测基准（Benchmark）和评测框架（Harness），以及构建私有评测集的方法。

---

## 1. 评测体系概述

### 1.1 评测的层次

```
┌────────────────────────────────────────────────────────────┐
│                    模型能力评测                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ 知识与   │  │ 推理与   │  │ 工具与   │  │ Agent     │  │
│  │ 常识     │  │ 逻辑     │  │ 代码     │  │ 能力      │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  基础模型 → SFT → RLHF/DPO → 工具学习 → Agent 训练    │ │
│  └──────────────────────────────────────────────────────┘ │
│                         ↑                                  │
│                      评测驱动                              │
└────────────────────────────────────────────────────────────┘
```

### 1.2 评测维度

| 维度 | 能力 | 典型任务 |
|------|------|----------|
| **知识** | 事实记忆、常识 | 问答、填空 |
| **推理** | 逻辑、数学、因果 | 数学题、逻辑推理 |
| **代码** | 代码生成、修复、解释 | HumanEval、MBPP |
| **对齐** | 安全性、指令遵循、有帮助 | RLHF benchmarks |
| **长上下文** | 文档理解、多跳推理 | NarrativeQA |
| **Agent** | 规划、工具调用、多步骤 | GAIA、WebArena |

---

## 2. 主流 Benchmark

### 2.1 通用能力

| Benchmark | 说明 | 特点 |
|-----------|------|------|
| **MMLU** | 57 个学科选择题 | 覆盖广，基础知识 |
| **BIG-Bench Hard (BBH)** | 23 个挑战性任务 | 需要深度推理 |
| **HELM** | 70+ 场景全面评测 | Holistic Evaluation |
| **SuperGLUE** | NLP 核心任务集合 | 继 GLUE 后继者 |
| **C-Eval** | 中文评测基准 | 覆盖 52 个学科 |
| **CMMLU** | 中文多任务理解 | 蒙藏等少数民族 |
| **AlignBench** | 中文对齐评测 | 安全性+有用性 |

### 2.2 推理能力

| Benchmark | 说明 |
|-----------|------|
| **GSM8K** | 8K 道小学数学题（8步以内） |
| **MATH** | 12K 道竞赛数学题 |
| **ARC** | 小学科学选择题（AI2） |
| **HellaSwag** | 常识推理（Vicuna 用） |
| **WinoGrande** | 常识代词消解 |

### 2.3 代码能力

| Benchmark | 说明 |
|-----------|------|
| **HumanEval** | OpenAI 164 道编程题 |
| **MBPP** | 974 道 Python 基础题 |
| **MultiPL-E** | 多语言代码生成 |
| **APPS** | 5000 道编程竞赛题 |
| **DS-1000** | Pandas/NumPy/SciPy 数据分析 |

### 2.4 中文能力

| Benchmark | 说明 |
|-----------|------|
| **C-Eval** | 中文专业知识 |
| **CMMLU** | 中文多任务 |
| **LCSTS** | 中文摘要生成 |
| **DuEE** | 中文事件抽取 |

---

## 3. lm-evaluation-harness 框架

### 3.1 概述

EleutherAI 出品的标准化评测框架，支持 60+ Benchmark，一行命令完成评测。

**GitHub**：`EleutherAI/lm-evaluation-harness`

### 3.2 安装

```bash
pip install lm-eval
```

### 3.3 基础用法

```bash
# 评测 gpt2 在 MMLU 上
lm_eval --model gpt2 \
  --model_args pretrained=gpt2 \
  --tasks mmlu \
  --device cuda:0

# 评测 llama 在多个任务上
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b \
  --tasks mmlu,arc_challenge,truthfulqa_mc,hellaswag \
  --batch_size 8
```

### 3.4 Python API

```python
from lm_eval import evaluate, tasks

# 加载模型
model = hf_model("meta-llama/Llama-2-7b")

# 评测
results = evaluate(
    model=model,
    tasks=["mmlu", "arc_challenge"],
    batch_size=8,
    limit=100  # 每任务只测 100 条（快速验证）
)

print(results)
```

### 3.5 添加自定义任务

**Task YAML 配置**：

```yaml
task: my_custom_task
dataset_path: my_org/my_dataset
dataset_name: null  # 用默认 split

# 输入模板
doc_to_text: "{{question}}"
doc_to_target: "{{answer}}"

# 解析方式
doc_to_choice: ["A", "B", "C", "D"]
process_results:
  my_metric:
    - function: exact_match
```

**注册并运行**：

```bash
lm_eval --model hf \
  --model_args pretrained=mymodel \
  --tasks my_custom_task \
  --include_path ./my_tasks
```

### 3.6 输出格式

```json
{
  "results": {
    "mmlu": {
      "acc": 0.65,
      "acc_stderr": 0.018
    },
    "arc_challenge": {
      "acc": 0.52
    }
  },
  "config": {
    "model": "meta-llama/Llama-2-7b",
    "batch_size": 8,
    "num_fewshot": 5
  }
}
```

---

## 4. HELM 评测体系

### 4.1 概述

Stanford CRFM 出品的 Holistic Evaluation of Language Models，覆盖 70+ 场景。

**GitHub**：`stanfordnlp/helm`

### 4.2 快速使用

```bash
pip install helms昊

# 运行评测
helm_run --models=anthropic claude-2 \
  --scenarios=mmlu,truthfulqa,bigbench \
  --num-train-trials=1

# 查看结果
helm_summarize --expire-days=7
```

### 4.3 核心场景

```python
scenarios = [
    "mmlu",           # 知识
    "truthfulqa",     # 真实性
    "realistic_toxicity",  # 安全性
    "bigbench",       # 综合推理
    "ms MARCO",       # 搜索排序
    "Leicester",       # 摘要
]
```

---

## 5. 构建私有评测集

### 5.1 评测集构建流程

```
确定评测目标 → 题目收集/生成 → 答案标注 → 质量审查 → 评测执行
     ↓                                           ↓
  定义能力维度                              统计分析与报告
```

### 5.2 题目来源

| 方法 | 优缺点 |
|------|--------|
| **人工撰写** | 质量高、针对性强，但成本高 |
| **从公开数据改写** | 成本低，但可能泄露 |
| **LLM 生成 + 人工审查** | 效率高，需要多轮质量把控 |
| **LLM 生成 + LLM 审查** | 可扩展，但可能循环偏差 |

### 5.3 题目类型

**选择题**：
```json
{
  "question": "某公司 2023 年毛利率为 30%，2024 年提升至 35%，\
主要原因是？",
  "options": [
    "A. 原材料成本下降",
    "B. 产品售价提升",
    "C. 产能利用率提升",
    "D. 以上皆有可能"
  ],
  "answer": "D",
  "explanation": "毛利率提升可由多重因素导致，需结合更多信息判断。"
}
```

**开放生成题**：
```json
{
  "question": "请分析以下案例中公司的竞争优劣势，并给出投资建议。",
  "reference": "应涵盖：1. 护城河分析 2. 财务健康度 3. 行业地位 4. 估值合理性",
  "rubric": {
    "优劣势分析": 40,
    "数据引用": 20,
    "逻辑清晰度": 20,
    "结论合理性": 20
  }
}
```

### 5.4 评估指标

**选择题**：
```python
accuracy = correct / total
balanced_accuracy = 平均每类准确率  # 处理类别不平衡
```

**生成题（LLM-as-Judge）**：
```python
# 用强模型对生成结果打分
def judge_score(response, rubric):
    prompt = f"""你是一位专业的评审。请根据以下评分标准对回答进行评分：

评分标准：{rubric}

回答：{response}

评分（0-100分）：
理由："""
    return parse_score(model.ask(prompt))
```

### 5.5 提示工程注意事项

```python
# 避免偏差：答案顺序对结果的影响
# A-B-C-D 顺序 vs D-C-B-A 顺序可能导致不同准确率

# 少样本选择：选什么例子影响模型表现
# 应选难度适中、覆盖不同类型的示例

# 评测时 temperature 应设为 0 或极低
# 避免随机性干扰评测稳定性
```

---

## 6. 评测最佳实践

### 6.1 评测设计原则

1. **避免数据泄露**：评测集与训练集不能有重叠
2. **足够样本量**：每个任务至少 300-1000 条，保证统计显著性
3. **区分难度层次**：简单/中等/困难分档评估
4. **可复现**：固定随机种子，控制 temperature = 0
5. **多维度**：不只看准确率，兼顾效率、成本、延迟

### 6.2 常见坑与避坑

| 坑 | 说明 | 避坑 |
|----|------|------|
| **数据泄露** | 评测集在训练时见过 | 用 held-out 测试集，每次重新划分子集 |
| **Eval-only 泄露** | 在公开评测上 Overfit | 提交前用私有评测验证 |
| **选择偏差** | 只报告好的任务 | 报告所有任务，不选择性披露 |
| **Prompt 敏感性** | 不同 Prompt 版本结果差异大 | 标准化 Prompt，并说明 Prompt 版本 |
| **随机性** | temperature > 0 导致波动 | 固定 seed，多次取平均 |

### 6.3 统计显著性

```python
from scipy.stats import ttest_rel
import numpy as np

# 同一评测集上两个模型的逐题对比
scores_a = [model_a.predict(q) for q in questions]
scores_b = [model_b.predict(q) for q in questions]

t_stat, p_value = ttest_rel(scores_a, scores_b)
if p_value < 0.05:
    print(f"模型 A 显著优于 B（p={p_value:.4f}）")
```

---

## 7. 主流评测框架对比

| 框架 | 支持任务 | 扩展性 | 学习曲线 |
|------|----------|--------|----------|
| **lm-eval (EleutherAI)** | 60+ | 高（YAML 配置） | 低 |
| **HELM (Stanford)** | 70+ | 中 | 中 |
| **BigCode-Evaluation** | 代码相关 | 中 | 低 |
| **OpenCompass (商汤)** | 中文场景丰富 | 高 | 低 |
| **FlagEval (智源)** | 中文基线 | 高 | 低 |

---

## 8. 参考资料

- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Stanford HELM](https://crfm.stanford.edu/helm)
- [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [BIG-Bench: Beyond the Imitation Game](https://arxiv.org/abs/2206.04615)
- [C-Eval: A Multi-Level Chinese Evaluation Benchmark](https://arxiv.org/abs/2305.08322)
