# Shiyan Benchmark

这个目录是在 [`../shiyan.md`](/home/minko/newswa/planC/shiyan.md) 的实验设想基础上，搭出来的一套 benchmark 工具层。底层依赖本地 vendored 的 `lm-evaluation-harness`，主要服务于 `flash-linear-attention` 里的 checkpoint，尤其是 `GM-SWA / SWA` 这一类模型。

README 这里尽量用中文说明实验目的、入口和注意事项，英文主要保留在任务名、脚本名、命令参数上，方便和 `lm-eval` 生态保持一致。

## 这套代码现在能做什么

- 支持真实 benchmark suite，而不是本地手搓 toy 数据
- 支持 `RULER`、`BABILong`、`LongBench v2`，以及部分 `LongBench` 任务
- 支持基于本地 checkpoint 的统一 harness runner
- 支持原来 `exp1 / exp2 / exp3 / exp4` 这套定制实验
- 支持结果落盘、CSV 聚合、简单画图
- 对 `GatedMemSWA` 提供 best-effort 的 trace 记录接口

## 目录结构

- `build_datasets.py`
  生成 suite manifest，或者在需要时生成旧版 demo JSONL
- `run_harness_suite.py`
  跑正式 benchmark suite 的主入口
- `run_eval.py`
  跑 `exp1 / exp2 / exp3 / exp4` 这类自定义实验
- `aggregate_results.py`
  把日志聚合成表格
- `plot_figures.py`
  画简单图表
- `shiyan_benchmark/`
  放通用 loader、evaluator、suite runner、trace 等实现

## 内置 Suite

### `paper_fast_v1`

用于日常迭代和 smoke check，但任务本身依然是真实 benchmark。

- `RULER` 子集
- `BABILong qa1-qa5`
- `LongBench v2` 的 history / code / table 等任务

特点：

- `RULER` 使用真实官方任务定义，但把 `num_samples` 降到 `100`，这样更适合频繁调参
- `BABILong` 仍然是真实数据集，不是模拟数据

### `paper_main_v1`

这是更接近论文主实验的正式套件。

- `RULER` 核心稳定任务
- `babilong_longctx`
- `LongBench v2` 核心任务
- 部分 `LongBench` appendix 风格任务

特点：

- 保留官方/默认规模，更适合最终表格和论文结果

### `mechanism_real_v1`

这是偏机制分析的真实数据套件，用于看 retrieval vs persistent 的区分。

- `retrieval_real`
  偏 `RULER` + `LongBench retrieval`
- `persistent_real`
  偏 `BABILong` 推理任务 + `LongBench v2` history/in-context 任务

## Quick Start

### 1. 生成真实 suite manifest

```bash
python /home/minko/newswa/planC/shiyan/build_datasets.py --mode real
```

生成后你会得到：

- [paper_fast_v1.json](/home/minko/newswa/planC/shiyan/data/splits/paper_fast_v1.json)
- [paper_main_v1.json](/home/minko/newswa/planC/shiyan/data/splits/paper_main_v1.json)
- [mechanism_real_v1.json](/home/minko/newswa/planC/shiyan/data/splits/mechanism_real_v1.json)

### 2. 跑真实 benchmark suite

先从 `paper_fast_v1` 开始最稳妥：

```bash
python /home/minko/newswa/planC/shiyan/run_harness_suite.py \
  --suite_file /home/minko/newswa/planC/shiyan/data/splits/paper_fast_v1.json \
  --model_name GMswa-40B-stage1 \
  --model_path /home/minko/newswa/planC/flash-linear-attention/flame/saves/GMswa-40B-stage1 \
  --import_module fla.models.gated_mem_swa \
  --device cuda \
  --dtype bfloat16 \
  --limit 1
```

如果你要正式跑论文主实验，把 `suite_file` 换成 `paper_main_v1.json` 即可。

### 3. 只在需要时生成旧版 demo 数据

这个模式主要是给最小化联调用的，不建议再把它当论文主数据源。

```bash
python /home/minko/newswa/planC/shiyan/build_datasets.py --mode demo
```

## 自定义实验入口

### 跑 `exp1`

```bash
python /home/minko/newswa/planC/shiyan/run_eval.py \
  --experiment exp1 \
  --model_name qwen2-gsw-128 \
  --model_path /home/minko/newswa/planC/flash-linear-attention/flame/saves/qwen2-gsw-128 \
  --task_file /home/minko/newswa/planC/shiyan/data/processed/task_decomposition_all.jsonl \
  --output_file /home/minko/newswa/planC/shiyan/results/logs/exp1_qwen2_gsw_128.jsonl \
  --import_module fla.models.gated_mem_swa \
  --config_override mem_gate_mode=linear \
  --config_override mem_proj_mode=linear \
  --device cuda \
  --dtype bfloat16
```

### 跑 `exp2`

```bash
python /home/minko/newswa/planC/shiyan/run_eval.py \
  --experiment exp2 \
  --model_name qwen2-gsw-128 \
  --model_path /home/minko/newswa/planC/flash-linear-attention/flame/saves/qwen2-gsw-128 \
  --task_file /home/minko/newswa/planC/shiyan/data/processed/minimal_recovery.jsonl \
  --output_file /home/minko/newswa/planC/shiyan/results/logs/exp2_qwen2_gsw_128.jsonl \
  --import_module fla.models.gated_mem_swa \
  --config_override mem_gate_mode=linear \
  --config_override mem_proj_mode=linear \
  --device cuda \
  --dtype bfloat16
```

### 聚合结果

```bash
python /home/minko/newswa/planC/shiyan/aggregate_results.py \
  --inputs /home/minko/newswa/planC/shiyan/results/logs/exp1_qwen2_gsw_128.jsonl \
  --experiment exp1
```

### 画图

```bash
python /home/minko/newswa/planC/shiyan/plot_figures.py --exp exp1
```

## 已经处理过的坑

### 1. 不再依赖本地模拟数据做主实验

`paper_fast_v1`、`paper_main_v1`、`mechanism_real_v1` 都是直接建立在本地 vendored `lm-evaluation-harness` 正式任务定义上的，不再是之前那种只适合 smoke test 的 toy 样例。

### 2. `GatedMemSWA` 生成 cache 已修

之前生成阶段会因为 `attn_state` 是 tuple、但 cache 更新逻辑把它当 list 写入而报错。这个兼容问题已经在 [utils.py](/home/minko/newswa/planC/flash-linear-attention/fla/models/utils.py) 修掉。

### 3. `BABILong` 多长度 metadata 已对齐

现在 `max_seq_lengths="0k,4k,16k,32k"` 这种写法会被正确解析，不再直接当成单个 `BuilderConfig` 名称。

实现位置在：

- [common_utils.py](/home/minko/newswa/planC/flash-linear-attention/flame/lm-evaluation-harness/lm_eval/tasks/babilong/common_utils.py)

### 4. 旧的 `.error.json` 不会一直残留

如果某次 run 失败，后面修好再重跑成功，旧的错误文件会自动清掉，避免结果目录里同时出现成功结果和过期错误。

## 使用建议

- 日常开发、调参数、先看链路通不通：
  用 `paper_fast_v1`
- 出论文主表、正式对比：
  用 `paper_main_v1`
- 做机制分析、ablation、memory 类型对比：
  用 `mechanism_real_v1`

## 注意事项

- `paper_main_v1` 保留的是更正式的 benchmark 规模，运行时间会明显更长
- `paper_fast_v1` 虽然更快，但仍然是正式 benchmark 子集，不是伪造样本
- `BABILong` 的多长度长上下文实验主要对 `qa1-qa5` 有意义
- `mechanism_real_v1` 里的 `qa11 / qa14 / qa15` 保持在 `0k`，这是按任务本身支持范围来定的
- benchmark 侧使用的是 `lm_eval.models.huggingface.HFLM`，但底层模型会先手动加载一遍，目的是让本地 `config_override` 更稳定
- trace logging 目前是 best-effort，主要针对 `GatedMemSWA` 的 gate 和 slot 信息

## 一个推荐工作流

1. 先用 `paper_fast_v1` 跑你当前 checkpoint，确认链路、显存和结果目录都正常。
2. 再用 `mechanism_real_v1` 看 retrieval / persistent 的差异是不是符合预期。
3. 最后再跑 `paper_main_v1`，产出论文主结果。
