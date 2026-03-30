
# GM-SWA 机制分析与主实验设计（Pretraining-Compatible 版本）

## 1. 文档目标

本文档用于指导 GM-SWA 论文中的实验设计，重点服务于以下三个目标：

1. 证明滑动窗口注意力（SWA）的窗口外退化，并不完全来自“远程 token 不可见”，而包含一类“持续影响型信息”的系统性流失。
2. 证明对这类信息而言，恢复一个紧凑长期状态即可保留主要预测效用，而不必显式重建远程 token 级访问。
3. 证明 GM-SWA 中的 `evicted value + gated memory` 不是任意 memory augmentation，而是与上述失效机制相匹配的设计。

本文档中的实验设计尽量采用 **pretraining-compatible** 的评测方式，即优先使用：

- next-token prediction
- cloze / multiple-choice scoring
- synthetic recall / consistency probes
- log-prob 对比
- teacher-forced 条件概率评测

尽量避免把核心机制分析建立在 instruction tuning 或开放式生成任务之上。

---

## 2. 总体实验结构

整篇实验建议分为两层：

### 第一层：机制分析（正文优先）
用于支撑引言中的问题发现与方法动机。

- 4.1 信息类型分解实验
- 4.2 最小恢复实验
- 4.3 记忆载体消融实验
- 4.4 门控写入机制分析与可视化

### 第二层：主结果实验
用于说明 GM-SWA 在标准预训练模型评测上的实际效果。

- Validation PPL
- RULER
- BABILong / InfiniteBench 子集
- 可选：LongBench 子集（建议放附录或补充）

---

## 3. 核心假设与实验对应关系

### 假设 H1
SWA 的窗口外退化并不完全来自远程 token 不可见，而至少包含一类“持续影响型信息”的系统性流失。

**对应实验：**
- 4.1 信息类型分解实验

### 假设 H2
对于持续影响型信息，恢复一个紧凑长期状态即可保留其主要预测效用，而不必显式恢复原始远程 token。

**对应实验：**
- 4.2 最小恢复实验

### 假设 H3
GM-SWA 中使用被逐出窗口的 value 作为长期记忆载体，是与问题结构匹配的，而不是任意 memory 设计都能达到类似效果。

**对应实验：**
- 4.3 记忆载体消融实验

### 假设 H4
持续影响型信息的保留本质上是一个“选择性保留”问题，因此门控写入优于无门控或固定衰减写入。

**对应实验：**
- 4.4 门控写入机制分析与可视化

---

## 4. 统一目录结构建议

project/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── annotations/
├── analysis/
│   ├── exp1_task_type_decomposition/
│   ├── exp2_minimal_recovery/
│   ├── exp3_memory_carrier_ablation/
│   └── exp4_gating_analysis/
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
├── scripts/
│   ├── build_datasets.py
│   ├── run_eval.py
│   ├── aggregate_results.py
│   └── plot_figures.py
└── paper_assets/
    ├── figure_captions.md
    └── table_notes.md



## 5. 统一数据记录格式

所有实验尽量统一存储格式，便于后续聚合、画图与论文写作。

### 5.1 基础结果表字段

```text
sample_id
task_name
task_group
model_name
context_length
window_size
condition
memory_type
write_type
score
metric_name
prediction
target
is_correct
extra_notes
```

### 5.2 字段说明

* `task_group`：

  * `retrieval`
  * `persistent`

* `condition`：

  * `default`
  * `original_span`
  * `summary_only`
  * `memory_only`
  * `none`

* `memory_type`：

  * `evicted_value`
  * `evicted_key`
  * `evicted_kv`
  * `window_summary`
  * `ema_hidden`
  * `random_history`
  * `none`

* `write_type`：

  * `gated`
  * `ungated`
  * `ema`
  * `overwrite`
  * `mean_pool`
  * `topk_write`
  * `no_write`

---

## 6. 实验 4.1：信息类型分解实验

### 6.1 实验目的

本实验旨在证明：窗口外依赖并非同质，至少可分为两类：

1. **精确检索型信息（retrieval-type）**

   * 后续步骤需要重新访问原始 token / span
   * 典型形式：变量名、数字、标识符、原文片段、表格单元格

2. **持续影响型信息（persistent-impact）**

   * 后续步骤不一定需要原始 token 本身，但需要保留其语义影响
   * 典型形式：任务约束、角色设定、实体属性、中间结论、状态更新

GM-SWA 的主要目标并不是全面替代远程 token 检索，而是补偿第二类信息。

---

### 6.2 数据构造

#### 6.2.1 Retrieval 组样本

每条样本应满足：

* 正确输出依赖一个较早出现的精确 span
* 当前评测位置时，该 span 已离开 SWA 窗口

可选来源：

* RULER retrieval 子任务
* Phonebook
* needle-style synthetic retrieval
* 代码符号远程补全
* 长上下文数字/实体精确回忆

推荐 JSONL 模板：

```json
{
  "sample_id": "retrieval_0001",
  "task_name": "needle_retrieval",
  "task_group": "retrieval",
  "context": "....",
  "choice_a": "The key is YX-2048.",
  "choice_b": "The key is YX-2408.",
  "label": "A",
  "evidence_span": {
    "start_char": 1820,
    "end_char": 1856,
    "text": "The access key is YX-2048."
  }
}
```

#### 6.2.2 Persistent 组样本

每条样本应满足：

* 早期信息会持续影响后续 continuation
* 后续不必逐 token 重读原 span
* 适合用 choice scoring 来评测

可选来源：

* 规则一致性
* persona 一致性
* 多轮状态跟踪
* 中间结论持续影响
* tool-return 状态一致性

推荐 JSONL 模板：

```json
{
  "sample_id": "persistent_0001",
  "task_name": "rule_consistency",
  "task_group": "persistent",
  "context": "....",
  "choice_a": "A train option under the budget would fit the user's needs.",
  "choice_b": "A seafood dinner and a flight would be ideal.",
  "label": "A",
  "persistent_factors": [
    "budget under 150",
    "no seafood",
    "avoid air travel"
  ],
  "evidence_span": {
    "start_char": 430,
    "end_char": 610,
    "text": "The user has a seafood allergy, limited budget, and prefers rail travel."
  }
}
```

---

### 6.3 评测方式

本实验不依赖开放式生成，而使用 **choice scoring**：

对两个候选 continuation 分别计算平均 token log-prob：

```text
score(choice) = mean_t log p(choice_t | context, choice_<t)
```

预测得分更高者为模型选择结果。

---

### 6.4 比较模型

主分析建议只保留：

* Full Attention
* SWA
* NSA（或一个强显式全局增强方法）
* GM-SWA

如果需要一个弱对照，可额外加入：

* Streaming-style sink retention

---

### 6.5 输出记录

结果表：
`results/tables/exp1_task_type_decomposition.csv`

字段建议：

```text
sample_id
task_name
task_group
model_name
score
metric_name
is_correct
context_length
window_size
```

---

### 6.6 聚合方式

按以下维度聚合：

* `task_group, model_name`
* `task_name, model_name`

统计：

* 平均准确率
* 平均 margin
* 相对 Full Attention 的保持率

保持率公式：

```text
retention = score(model) / score(full_attention)
```

---

### 6.7 图表设计

#### 图 2：不同窗口外信息类型上的性能比较

* 横轴：`retrieval` / `persistent`
* 纵轴：平均准确率或归一化得分
* 柱子：SWA / NSA / GM-SWA / Full Attention

#### 图注建议

在精确检索型与持续影响型任务上比较不同模型的平均表现。结果显示，GM-SWA 在持续影响型任务上相较纯 SWA 有显著提升，并接近显式全局增强方法；而在严格依赖原始 span 检索的任务上，显式全局增强仍更具优势。

---

### 6.8 预期结果与论文分析要点

预期现象：

* 在 retrieval 组中，显式全局增强通常优于 GM-SWA
* 在 persistent 组中，GM-SWA 显著优于纯 SWA，并接近显式全局增强

论文中的分析要点：

1. 窗口外依赖并非同质
2. 持续影响型信息是一类独立的长期依赖形式
3. 对这类信息而言，紧凑长期状态是合理建模对象

---

## 7. 实验 4.2：最小恢复实验

### 7.1 实验目的

本实验直接回答：

> 对已经离开窗口的关键历史信息，模型是否必须恢复原始 span，还是恢复一个紧凑长期状态就足够？

这是全文最关键的机制分析实验之一。

---

### 7.2 样本构造

从实验 4.1 的样本中筛选出一批可明确标注关键 span 的样本。
筛选条件：

* 关键 span 对当前 continuation 有明显影响
* 当前时刻时该 span 已经离开 SWA 窗口
* span 长度控制在 10–80 token

---

### 7.3 四种条件构造

对每个样本构造 4 个版本：

#### 条件 1：Original

原始关键 span 保留在上下文中，可显式访问。

#### 条件 2：Summary

删除原始关键 span，用简短文本摘要替代。

示例：

* 原 span：`The user is allergic to peanuts, avoids seafood, and has a limited budget.`
* summary：`Constraints: peanut allergy, no seafood, low budget.`

#### 条件 3：Memory-style

删除原始关键 span，替换为更抽象、更紧凑的状态化描述。

示例：

* `State: peanut_allergy=true; seafood=false; budget=low`

#### 条件 4：None

删除原始关键 span，且不给任何恢复信息。

---

### 7.4 数据格式

推荐 JSONL 模板：

```json
{
  "sample_id": "persistent_0021",
  "task_group": "persistent",
  "task_name": "constraint_continuation",
  "context_original": "....",
  "context_summary": "....",
  "context_memory": "....",
  "context_none": "....",
  "target_continuation": "A train option under 150 dollars would fit the user's needs."
}
```

---

### 7.5 评测方式

固定目标 continuation，分别计算其在四个条件下的 log-prob：

* `log p(target | original)`
* `log p(target | summary)`
* `log p(target | memory)`
* `log p(target | none)`

可记录：

* 平均 token log-prob
* perplexity on target continuation
* 相对 none 的恢复增益

恢复增益定义：

```text
delta_vs_none = score(condition) - score(none)
```

---

### 7.6 输出记录

结果表：
`results/tables/exp2_minimal_recovery.csv`

字段建议：

```text
sample_id
task_group
task_name
model_name
condition
score
metric_name
delta_vs_none
```

---

### 7.7 图表设计

#### 图 3(a)：Retrieval 任务上的最小恢复结果

#### 图 3(b)：Persistent 任务上的最小恢复结果

* 横轴：None / Summary / Memory / Original
* 纵轴：准确率或平均 log-prob

#### 图注建议

对已经离开窗口的关键 span，分别提供原始远程访问、压缩摘要、状态化摘要和无恢复四种设置。结果表明，在持续影响型任务上，摘要与状态化恢复均可显著弥补性能下降，并在多项任务上接近原始远程访问；而在精确检索型任务上，恢复原始 span 仍是最有效方式。

---

### 7.8 预期结果与论文分析要点

预期现象：

* retrieval 组中：`Original > Summary ≈ Memory > None`
* persistent 组中：`Original ≈ Summary ≈ Memory > None`

论文中的分析要点：

1. 并非所有窗口外依赖都需要原始 token 级访问
2. 对 persistent 组，恢复压缩后的状态已经足够
3. 这直接支持“长期影响保留”而非“远程 token 检索”这一问题定义

---

## 8. 实验 4.3：记忆载体消融实验

### 8.1 实验目的

本实验用于回答：

> 为什么 GM-SWA 要把被逐出窗口的 value 写入长期记忆？
> 是否任意 memory 载体都能带来类似提升？

---

### 8.2 记忆载体配置

保持 memory 大小、读出方式、训练设置一致，仅改变写入载体：

* `evicted_value`
* `evicted_key`
* `evicted_kv`
* `window_summary`
* `ema_hidden`
* `random_history`
* `none`

---

### 8.3 每种载体定义建议

#### evicted_value

当前窗口前移时刚被淘汰的 value

#### evicted_key

刚被淘汰的 key

#### evicted_kv

对淘汰的 K 和 V 做 joint projection 或 concat 后投影

#### window_summary

对当前窗口内所有 value 做平均或加权平均

#### ema_hidden

对历史 hidden states 做 EMA

#### random_history

从历史位置随机抽取一个 value 或 span summary

#### none

不使用长期记忆

---

### 8.4 评测任务

建议仅选以下三类，避免计算量过大：

1. Validation PPL
2. Retrieval probe（如 RULER retrieval 子任务）
3. Persistent probe（实验 4.1 中 persistent 组）
4. 可选：LongBench average 或一个长上下文综合指标

---

### 8.5 输出记录

结果表：
`results/tables/exp3_memory_carrier_ablation.csv`

字段建议：

```text
model_name
memory_source
task_name
task_group
score
metric_name
ppl
extra_params
extra_latency
```

---

### 8.6 表格设计

#### 表 2：不同记忆写入载体的消融比较

| 写入载体 | Val PPL ↓ | Retrieval Acc ↑ | Persistent Acc ↑ | Avg Margin ↑ | Extra Latency |
| ---- | --------: | --------------: | ---------------: | -----------: | ------------: |

---

### 8.7 预期结果与论文分析要点

预期现象：

* `evicted_value` 最稳，特别是在 persistent 组上
* `random_history` 和 `ema_hidden` 明显较弱
* `evicted_key` 不一定优于 `evicted_value`
* `evicted_kv` 可能略强或相近，但复杂度更高

论文中的分析要点：

1. 不是任意历史摘要都能有效补偿 SWA 的窗口外退化
2. value 更接近“当前层已编码的语义输出”，适合作为长期影响载体
3. GM-SWA 的目标不是可逆检索，而是长期语义影响保留

---

## 9. 实验 4.4：门控写入机制分析与可视化

### 9.1 实验目的

本实验用于验证：

> 持续影响型信息的保留是否本质上是一个“选择性保留”问题？
> 如果是，那么 gated write 是否优于无门控或固定衰减写入？

---

### 9.2 写入方式配置

建议至少比较以下 6 种：

* `gated`
* `ungated`
* `ema`
* `overwrite`
* `mean_pool`
* `no_write`

可选补充：

* `topk_write`

---

### 9.3 每种写入方式定义

#### gated

原始 GM-SWA 设计，门控系数由当前输入动态决定

#### ungated

固定系数混合，例如：

```text
m_t = alpha * m_{t-1} + (1 - alpha) * proj(v_evict)
```

#### ema

标准指数滑动平均

#### overwrite

直接覆盖部分 slot，不做保留控制

#### mean_pool

做简单累积平均

#### no_write

保留 memory 分支结构，但不更新 memory

#### topk_write（可选）

仅在显著性分数较高时更新 memory

---

### 9.4 评测任务

优先使用：

* persistent probe
* retrieval probe
* validation PPL

因为本实验主要为“持续影响型信息选择性保留”服务，不必全 benchmark 覆盖。

---

### 9.5 输出记录

结果表：
`results/tables/exp4_gating_ablation.csv`

字段建议：

```text
write_type
task_name
task_group
score
metric_name
ppl
extra_latency
memory_utilization
slot_sparsity
```

---

### 9.6 可视化日志记录

在推理时对每一步记录：

* `sample_id`
* `step_id`
* `slot_id`
* `gate_value`
* `slot_norm`
* `read_weight`
* `event_type`
* `event_just_evicted`

保存文件：
`results/logs/exp4_slot_trace_case_xxx.csv`

字段模板：

```text
sample_id
step_id
slot_id
gate_value
slot_norm
read_weight
event_type
event_just_evicted
```

---

### 9.7 可视化设计

#### 图 4：关键历史事件离开窗口后记忆槽激活的变化

推荐两种画法：

##### 方案 A：Heatmap

* 横轴：time step
* 纵轴：slot id
* 颜色：gate value 或 slot norm
* 在关键事件离开窗口时刻画竖线

##### 方案 B：Line Plot

* 选若干活跃 slot
* 画 gate value / slot norm 随时间变化曲线
* 标注关键事件离开窗口的时刻

---

### 9.8 预期结果与论文分析要点

预期现象：

* `gated` 在 persistent 组上表现最好
* `ungated` 和 `ema` 有提升，但不如 `gated`
* 在关键约束、状态变化、中间结论离开窗口后，某些 slot 的激活明显上升
* retrieval 组中，这种现象通常不如 persistent 组显著

论文中的分析要点：

1. 持续影响型信息的保留不是“越多越好”，而是“哪些值得保留”
2. gate 机制不是装饰，而是执行选择性 retention 的关键部件
3. GM-SWA 不是一般性记忆缓存，而是在做与问题结构一致的选择性长期保留

---

## 10. 主结果实验（Pretraining-Compatible）

### 10.1 目标

在机制分析之外，验证 GM-SWA 在标准预训练模型评测中的综合效果。

---

### 10.2 推荐主评测集

#### 基础建模能力

* Validation PPL（如 FineWeb-edu / C4 / Pile held-out）

#### 精确检索型长上下文任务

* RULER
* Phonebook
* Needle-style retrieval
* Key-value recall

#### 长上下文综合任务

* BABILong
* InfiniteBench 子集（优先选择不强依赖 instruction alignment 的子任务）

#### 可选附录

* LongBench / LongBench v2 子集

---

### 10.3 主比较模型建议

正文主表建议不超过 5 个：

* Full Attention
* SWA
* NSA
* GSA（他是带有 softmax 的 linear attention）
* GM-SWA

如果担心模型太多，可将 GSA 放到附表或分析表中。

---

### 10.4 主结果表建议

#### 表 1：主实验结果

| Model | Val PPL ↓ | RULER Avg ↑ | Retrieval Avg ↑ | Persistent Avg ↑ | Decode Memory ↓ | Throughput ↑ |
| ----- | --------: | ----------: | --------------: | ---------------: | --------------: | -----------: |

---

## 11. 统一实现建议

### 11.1 数据构造脚本

脚本：
`build_datasets.py`

输出文件建议：

```text
data/processed/task_decomposition_retrieval.jsonl
data/processed/task_decomposition_persistent.jsonl
data/processed/minimal_recovery.jsonl
```

---

### 11.2 统一评测脚本

脚本：
`run_eval.py`

示例命令：

```bash
python run_eval.py \
  --model_name gmswa \
  --task_file data/processed/task_decomposition_persistent.jsonl \
  --window_size 1024 \
  --condition default \
  --memory_source evicted_value \
  --write_type gated \
  --output_file results/logs/run_xxx.jsonl
```

---

### 11.3 统一聚合脚本

脚本：
`aggregate_results.py`

功能：

* 从原始 run 日志中提取样本分数
* 合并为统一 CSV
* 对 task / group / model 做聚合
* 计算均值、方差、相对保持率、margin

输出建议：

```text
results/tables/exp1_task_type_decomposition.csv
results/tables/exp2_minimal_recovery.csv
results/tables/exp3_memory_carrier_ablation.csv
results/tables/exp4_gating_ablation.csv
```

---

### 11.4 统一作图脚本

脚本：
`plot_figures.py`

示例命令：

```bash
python plot_figures.py --exp exp1
python plot_figures.py --exp exp2
python plot_figures.py --exp exp4 --case_id persistent_014
```

---

## 12. 论文中各实验与方法设计的对应关系

为避免方法设计显得任意，正文中应明确建立如下对应关系：

* **引言中的问题发现**

  * 对应实验 4.1 与实验 4.2

* **方法中为什么写 evicted value**

  * 对应实验 4.3

* **方法中为什么要使用 gate**

  * 对应实验 4.4

这样可以保证：

* 每个设计点都有实验支撑
* 不是“先设计方法，再补故事”
* baseline 选择也会自然收敛

---

## 13. 推荐执行优先级

若资源有限，建议按以下顺序执行：

### 第一优先

**实验 4.2：最小恢复实验**

* 最直接证明“不是所有窗口外依赖都需要原始 token retrieval”

### 第二优先

**实验 4.3：记忆载体消融**

* 最直接证明“为什么是 evicted value”

### 第三优先

**实验 4.1：信息类型分解**

* 最能帮助整篇论文收敛动机与 baseline 边界

### 第四优先

**实验 4.4：门控分析与可视化**

* 最能增强说服力与 reviewer 观感

---

## 14. 论文写作时可直接使用的阶段性结论

### 结论 1

窗口外依赖并非同质，至少可以区分为精确检索型与持续影响型两类。

### 结论 2

对于持续影响型信息，恢复一个紧凑长期状态已足以弥补纯 SWA 的大部分性能下降，而不必显式重建远程 token 级访问。

### 结论 3

被逐出窗口的 value 是更适合承载持续影响型信息的长期状态载体。

### 结论 4

门控写入是选择性保留持续影响型信息的关键机制，而非一般性的附加模块。

---

## 15. 最终建议的正文实验组织

推荐实验章节结构如下：

### 4.1 滑动窗口注意力窗口外信息丢失的机制分析

* 4.1.1 信息类型分解
* 4.1.2 最小恢复实验
* 4.1.3 记忆载体消融
* 4.1.4 门控写入分析与可视化

### 4.2 主实验结果

* Validation PPL
* RULER
* Retrieval probes
* Persistent probes

### 4.3 与显式全局增强方法和紧凑记忆方法的比较

* SWA
* NSA
* GSA
* GM-SWA

### 4.4 迁移适配与效率分析

* 从 SWA checkpoint 转到 GM-SWA
* Decode memory / throughput
* 可选：prefix scan 效率分析

---

## 16. 一句话总结

本实验设计的核心思想是：

**先证明 SWA 窗口外丢失的不是单一的“远程 token 可见性”，而是包含一类“持续影响型信息”的系统性流失；再证明这类信息可以通过紧凑长期状态恢复；最后证明 GM-SWA 中的 evicted-value gated memory 恰好是与这一问题结构相匹配的低成本补偿机制。**

