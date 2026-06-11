# 协助训练的说明

这次需要帮忙在 FineWeb-Edu-100B 上训练并评测 GMSWA 及 5 个 baseline：`gated_mem_swa`、`swa`、`transformer`、`gated_deltanet`、`gsa`、`nsa`，每个模型跑 `340M` 和 `1B` 两个规模。默认使用 8 张 H100/A100 80GB，直接按下面 runbook 的 `scripts/run_all.sh` 或 `scripts/run_one.sh <model> <scale>` 启动；smoke test：

```bash
source .venv311/bin/activate
CUDA_VISIBLE_DEVICES=4 NGPU=1 SKIP_EVAL=1 WANDB=0 \
STEPS=3 SEQ_LEN=16384 GRAD_ACCUM=1 CKPT_INTERVAL=1000 \
DATASET=/home/user01/Minko/datasets/fineweb_edu_100BT \
TOKENIZER=/home/user01/Minko/models/gla-tokenizer \
bash scripts/run_one.sh gated_mem_swa 340M
```

注意几点：

- 请使用 Python 3.11 + torch 2.8 + `flash-attn==2.8.3`。
- `gated_mem_swa` 和 `swa` 是最关键的一组对比，其中 `swa` 是关闭 memory 的严格 ablation。资源紧张时，优先跑 `gated_mem_swa` vs `swa`，再补其它 baseline。
- 每个 run 的训练日志、转换日志和 eval 结果会写到 `eval_results/<model>-<scale>/`，checkpoint/HF 权重会写到 `flash-linear-attention/flame/saves/<model>-<scale>/`。

---

# GMSWA — experiments runbook

Train and evaluate GMSWA against five baselines (SWA, transformer, gated_deltanet,
GSA, NSA) at two scales (340M, 1B), all on FineWeb-Edu-100B with matched
hyper-parameters.

```
340M  →  50B tokens   (50_000 steps × ~1M tokens/step on 8 GPUs)
1B    → 100B tokens   (100_000 steps × ~1M tokens/step on 8 GPUs)
```

Eval is `lm-eval`'s short-context suite + a few long-context tasks.

---

## 1. Environment

Hardware: a single node with **8 × H100 80GB** (or A100 80GB).
Software: CUDA 12.x driver, Python 3.11, `uv` package manager.

```bash
# install uv once if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone (the repo is large; ssh recommended)
cd /home/<you>/
git clone <repo-url> GMSWA
cd GMSWA

# one-shot installation that pins every package to the exact version we used
bash scripts/setup_env.sh
source .venv311/bin/activate
```

Pinned versions live in `requirements.lock.txt` (`uv pip freeze` of the working
env). Key versions:

| package        | version |
| -------------- | ------- |
| python         | 3.11.x  |
| torch          | 2.8.0   |
| triton         | 3.4.0   |
| transformers   | 4.57.0  |
| lm-eval        | 0.4.11  |
| accelerate     | 1.13.0  |
| datasets       | 4.5.0   |
| flash-attn     | 2.8.3   |
| wandb          | 0.25.1  |
| torchtitan     | git@`0b44d4c` |

**flash-attn caveat.** Use a wheel/build that matches Python 3.11 and torch 2.8.
On our lab box this is `flash-attn==2.8.3`. Transformer, NSA, and packed-varlen
GMSWA training need a working flash-attn varlen kernel.

```bash
uv pip install flash-attn==2.8.3 --no-build-isolation
```

### Data and tokenizer (already on disk on the lab box)

```
/home/user01/Minko/datasets/fineweb_edu_100BT      <- pretraining corpus
/home/user01/Minko/models/gla-tokenizer            <- shared tokenizer (vocab 32k)
```

If you move them, set `DATASET=/path` and `TOKENIZER=/path` env vars when
invoking the scripts.

---

## 2. Models and configs

All configs live in `flash-linear-attention/flame/configs/`. Six model families ×
two scales = 12 configs. Param counts are body+embedding total.

| nickname        | 340M file               | 1B file               | 340M params | 1B params | notes |
| --------------- | ----------------------- | --------------------- | -----------:| ---------:| ----- |
| `gated_mem_swa` | `gated_mem_swa_340M.json` | `gated_mem_swa_1B.json` | 337M | 1.22B | **ours** (SWA + TTT-style fast weights) |
| `swa`           | `swa_340M.json`         | `swa_1B.json`         | 336M | 1.21B | strict SWA baseline (= GMSWA with `disable_memory=true`) |
| `transformer`   | `transformer_340M.json` | `transformer_1B.json` | ~407M | ~1.4B | full attention reference |
| `gated_deltanet`| `gated_deltanet_340M.json` | `gated_deltanet_1B.json` | 512M | 1.39B | linear-attention reference |
| `gsa`           | `gsa_340M.json`         | `gsa_1B.json`         | 380M | 1.38B | softmax-linear hybrid reference |
| `nsa`           | `nsa_340M.json`         | `nsa_1B.json`         | 382M | 1.40B | DeepSeek native sparse attention |

GMSWA's `disable_memory=true` flag is what makes `swa` a *strict* ablation —
same projections, same window, same RoPE, only the gated fast-weight memory
path is off. Use this baseline for any "memory ablation" claim.

---

## 3. Running it

All three scripts live in `scripts/`. Output goes to `eval_results/<run>/...` and
checkpoints to `flash-linear-attention/flame/saves/<run>/`.

### 3a. Train + eval everything (12 runs, sequential)

```bash
bash scripts/run_all.sh
```

It loops `for scale in 340M 1B: for model in <6 models>: train, convert DCP→HF,
run lm-eval`. Each run writes its own log; the queue log is at
`eval_results/queue.log`. A failed run does NOT abort the others (set
`STOP_ON_ERROR=1` to change that).

After the queue finishes, `eval_results/summary.csv` has one row per
(run, task, metric).

### 3b. Subset of runs

```bash
# all 340M only
bash scripts/run_all.sh --scales 340M

# just our model vs strict-SWA at both scales
bash scripts/run_all.sh --models gated_mem_swa,swa

# only the 1B GMSWA run
bash scripts/run_all.sh --models gated_mem_swa --scales 1B
```

### 3c. Single run (interactive use / debugging)

```bash
bash scripts/run_one.sh gated_mem_swa 340M
bash scripts/run_one.sh transformer   1B
```

Useful env vars for `run_one.sh` / `run_all.sh`:

| var | default | effect |
| --- | ------- | ------ |
| `NGPU`           | 8 | how many GPUs to use (data_parallel_shard_degree) |
| `SKIP_TRAIN=1`   | 0 | skip training, only convert + eval |
| `SKIP_EVAL=1`    | 0 | train only, no eval |
| `STOP_ON_ERROR=1`| 0 | (run_all only) abort the queue on first failure |
| `WANDB=1`        | 1 | toggle wandb logging |
| `DATASET=...`    | — | override dataset path |
| `TOKENIZER=...`  | — | override tokenizer path |

### 3d. Eval only (model already trained)

```bash
bash scripts/eval_one.sh gated_mem_swa-340M \
     flash-linear-attention/flame/saves/gated_mem_swa-340M \
     eval_results/gated_mem_swa-340M
```

`SHORT_TASKS` / `LONG_TASKS` env vars override the default task lists.

---

## 4. Hyper-parameters (frozen across all 6 models at each scale)

|                      | 340M    | 1B      |
| -------------------- | -------:| -------:|
| total tokens         |   50B   |  100B   |
| steps                | 50 000  | 100 000 |
| learning rate        | 7e-4    | 1e-3    |
| lr_min               | 7e-5    | 1e-4    |
| warmup steps         | 5 000   | 10 000  |
| decay                | cosine, decay_ratio 0.2 | same |
| optimizer            | AdamW, eps 1e-15 | same |
| `seq_len` (varlen)   | 65 536  | 16 384  |
| grad accum           | 2       | 8       |
| GPUs (`dp_shard`)    | 8       | 8       |
| tokens / step        | 1 048 576 (≈1M) | 1 048 576 (≈1M) |
| ckpt interval        | 5 000   | 10 000  |

`seq_len × grad_accum × NGPU = 2²⁰ = 1M tokens/step` is the invariant we keep
constant across scales.

---

## 5. Evaluation suite

Short-context (default in `eval_one.sh`):

```
piqa, openbookqa, hellaswag, arc_easy, arc_challenge, wikitext
```

Long-context:

```
longbench_hotpotqa, longbench_qasper, niah_single_2
```

Override with `SHORT_TASKS="..." LONG_TASKS="..."`. Long-context eval runs with
`max_length=8192` and batch size 1 to avoid OOM; bump `max_length` if you want
to test 16k/32k.

---

## 6. Output layout

```
eval_results/
├── queue.log                       # high-level driver log (run_all only)
├── summary.csv                     # aggregated metrics, one row per (run,task,metric)
├── <model>-<scale>/
│   ├── train.log                   # full training stdout (tee'd)
│   ├── convert.log                 # DCP → HF conversion log
│   ├── short.json                  # lm-eval raw output (short tasks)
│   ├── short.log
│   ├── long.json                   # lm-eval raw output (long-context tasks)
│   └── long.log
└── ...

flash-linear-attention/flame/saves/
├── <model>-<scale>/
│   ├── checkpoint/                 # DCP shards during training
│   ├── config.json                 # HF config after conversion
│   ├── *.safetensors               # consolidated weights
│   └── tokenizer.*                 # tokenizer copy
```

Re-running aggregation only:

```bash
python scripts/aggregate_eval.py --eval-root eval_results --out eval_results/summary.csv
```

---

## 7. Estimated wall-clock (8 × H100)

Rough budget for planning, not contractual:

| run                 | tokens | est. wall-clock |
| ------------------- | ------:| ---------------:|
| `*-340M`            |   50B  |    ~3 days each |
| `*-1B`              |  100B  |   ~10 days each |

Six 340M runs ≈ **18 days** of GPU-time; six 1B runs ≈ **60 days** if run
sequentially. Plan parallelism (multiple nodes) accordingly — `run_all.sh` is
sequential by design; for parallel training across nodes, launch
`run_one.sh <model> <scale>` on separate nodes.

---

## 8. What lives where

| path | purpose |
| ---- | ------- |
| `flash-linear-attention/fla/layers/gated_mem_swa.py` | GMSWA v2 layer (TTT-style fast weights) |
| `flash-linear-attention/fla/models/gated_mem_swa/` | HF integration |
| `flash-linear-attention/flame/configs/*.json` | training configs (all baselines + ours) |
| `flash-linear-attention/flame/train.sh` | torchrun launcher |
| `flash-linear-attention/flame/eval.py` | legacy eval entry (now superseded by `scripts/eval_one.sh`) |
| `scripts/setup_env.sh` | one-shot environment install |
| `scripts/run_one.sh` | train + convert + eval one (model, scale) |
| `scripts/run_all.sh` | driver over all (model, scale) combos |
| `scripts/eval_one.sh` | lm-eval suite runner |
| `scripts/aggregate_eval.py` | collect all eval JSONs into one CSV |
| `paper/gmswa_v2_design.md` | architectural design doc for GMSWA v2 |
| `test_gmswa_v2.py` | unit tests for the GMSWA layer (run on any single GPU) |
| `requirements.lock.txt` | full `uv pip freeze` of the working env |

---

## 9. Quick sanity checks before kicking off a 10-day run

```bash
# verify env
python -c "import torch, fla, lm_eval; print(torch.cuda.is_available(), fla.__name__, lm_eval.__version__)"

# verify all 12 configs load
python - <<'PY'
import sys; sys.path.insert(0,"flash-linear-attention"); import fla
from transformers import AutoConfig
for s in ("340M","1B"):
    for m in ("gated_mem_swa","swa","transformer","gated_deltanet","gsa","nsa"):
        c = AutoConfig.from_pretrained(f"flash-linear-attention/flame/configs/{m}_{s}.json")
        print(f"  {m:<18}{s:<5} -> model_type={c.model_type}")
PY

# small-model unit tests for GMSWA (~30s on one GPU)
CUDA_VISIBLE_DEVICES=4 python test_gmswa_v2.py

# dry-run one config end-to-end with 100 steps (smoke train + eval pipeline)
SKIP_EVAL=1 bash scripts/run_one.sh gated_mem_swa 340M  # then Ctrl-C after a few steps
```

---

## 10. Common gotchas

- **`flash_attn_2_cuda.so: undefined symbol`** — bundled wheel ABI mismatch with
  installed torch. GMSWA / SWA still run (SDPA fallback). For
  `transformer` / `nsa`, rebuild flash-attn from source.
- **`BuilderConfig '0k,4k,16k,32k' not found`** — older babilong cache; use the
  task names listed in `LONG_TASKS` instead.
- **DCP → HF conversion fails after training** — check `<save_dir>/checkpoint/`
  for shard files; if step number is off, pass `--step <N>` manually to
  `python -m flame.utils.convert_dcp_to_hf`.
- **wandb prompts for login** — `wandb login` once, or set `WANDB=0`.
- **OOM on long-context eval** — drop `max_length` in `eval_one.sh` or use
  fewer tasks.
