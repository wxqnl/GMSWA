# GMSWA — Gated-Memory Sliding-Window Attention

A constant-cache sparse-attention layer: exact sliding-window attention (SWA) fused,
through a learned per-head gate, with a gated-delta-rule **recurrent memory** (induction
short-conv + NoPE retrieval projections). The KV cache is **independent of context length**.

This repo contains the implementation, the controlled 340M study, the long-context
efficiency benchmark, and the AAAI paper draft.

> The original training runbook (env-var driven `run_one.sh`/`run_all.sh`) is preserved at
> `scripts/archive/old_training_runbook.md`.

---

## TL;DR — what we found (340M, controlled, parameter-matched)

| Axis | GMSWA vs. baselines |
|---|---|
| **Base quality** (zero-shot acc, loss-vs-position) | **> GDN** (recurrent); on par with softmax baselines |
| **Real recall** (SWDE/FDA/SQuAD) | **comparable to GDN** (wins SQuAD/FDA, trails SWDE) |
| **Efficiency** | **constant** KV cache — **~790× smaller** than full attention at 128K |
| **Synthetic single-needle recall** (NIAH) | ≈ SWA, **trails GDN** — the honest limitation |

The headline is a **negative result with analysis**: five interventions (output gated-norm,
two pathway-dropout strengths, a memory-first curriculum) all fail to give the hybrid sharp
single-needle recall. The window handles local prediction, so the recurrent memory learns
*smooth/semantic* recall, not *sharp synthetic* retrieval — a "smooth-vs-sharp recall"
design caution for window–memory hybrids. Full write-up: `paper/GMSWA_final.md`.

---

## Repository structure

```
GMSWA/
├── readme.md                     ← this file
├── paper/                        ← the AAAI paper (final, reviewed)
│   ├── GMSWA_AAAI_final.tex/.pdf  ← compilable AAAI submission (clean build)
│   ├── GMSWA_final.md             ← readable markdown source
│   ├── RESULTS_DOSSIER.md         ← ground-truth numbers for every claim
│   ├── references.bib             ← citations (verify arXiv IDs before submission)
│   ├── TODO_experiments.md        ← deferred (GPU-gated) experiments, prioritized
│   ├── figures/efficiency.{pdf,png}
│   └── archive/                   ← superseded earlier drafts
├── scripts/
│   ├── analysis/                  ← analyze_suite.py, ppl_vs_position.py
│   ├── benchmark/                 ← speed_benchmark_v2.py (long-context efficiency)
│   ├── probes/                    ← mechanism experiments (capacity, MQAR, toy recall)
│   ├── tests/                     ← test_mem_conv_consistency.py (prefill/decode checks)
│   ├── archive/                   ← dead/superseded exploratory scripts + old runbook
│   ├── lm_eval_fla.py             ← lm-eval shim for fla models
│   └── run_one.sh / run_all.sh / setup_env.sh
├── flash-linear-attention/
│   ├── fla/layers/gated_mem_swa.py         ← the GMSWA layer (flags below)
│   ├── fla/models/gated_mem_swa/           ← config + HF model
│   └── flame/                              ← torchtitan training launchers
│       ├── configs/*.json                  ← one config per model / ablation
│       ├── run_{gmswa_v5conv,gdn,swa,transformer}_340M_10k.sh   ← canonical training
│       ├── run_gmswa_memonly_340M.sh        ← memory-only ablation (experiment A, ready)
│       ├── run_gmswa_v3_1B.sh               ← 1B (deferred)
│       ├── eval_suite.sh                    ← NIAH sweep + real-recall eval
│       ├── run_eval_{v5conv,memonly}.sh     ← convert→ppl→NIAH→recall→short-bench chain
│       └── archive/                         ← done ablations + one-off orchestration
└── eval_results/                 ← all eval outputs (suite/, ppl_*.out, */short/)
```

`fla/layers/gated_mem_swa.py` config flags (all opt-in, backward-compatible):
`disable_memory`, `disable_local` (memory-only ablation), `mem_separate_proj`,
`mem_evicted_only`, `mem_use_short_conv`, `mem_use_output_norm`,
`mem_swa_drop_prob` (+ `mem_swa_drop_anneal_steps` for the curriculum).

### The "GMSWA (ours)" final config
`v5conv` = SWA (W=512) + **full-sequence** gated-delta memory with a **short-conv** and
NoPE retrieval (`mem_evicted_only:false`, `mem_use_short_conv:true`). The ablation ladder
(`v6` output-norm, `v7/v8` dropout, `v9` curriculum) lives in `flame/archive/` and
`flame/configs/`; all were strictly worse on recall (see `paper/RESULTS_DOSSIER.md`, Table 6).

---

## Environment

- Single node, **8 × H100/A100 80GB**. Python **3.11**, torch **2.8**, `flash-attn==2.8.3`.
- venv: `.venv311` (`source .venv311/bin/activate`).
- Data: `/home/user01/Minko/datasets/fineweb_edu_100BT` (local, streamed).
- Tokenizer: `/home/user01/Minko/models/gla-tokenizer` (32K vocab).

---

## Reproduce the 340M study

All models: 340M (≈369M), 24 layers, d=1024, 10K steps, ctx 2048, identical recipe.

```bash
cd flash-linear-attention/flame
# train GMSWA + the three baselines (each ~4.5h on 8 GPUs)
bash run_gmswa_v5conv_340M.sh      # GMSWA (SWA + full-seq gated-delta memory + short-conv)
bash run_gdn_340M_10k.sh           # gated DeltaNet (recurrent baseline)
bash run_swa_340M_10k.sh           # sliding-window-only ablation
bash run_transformer_340M_10k.sh   # full-attention baseline
# evaluate each (convert DCP→HF, then NIAH sweep + real recall + zero-shot)
bash run_eval_v5conv.sh            # template; clone per model
```
Aggregate the master tables:
```bash
cd /home/user01/Minko/GMSWA
python scripts/analysis/analyze_suite.py SWA Transformer GMSWA-v5conv GDN
```

### Long-context efficiency figure
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=flash-linear-attention \
  .venv311/bin/python scripts/benchmark/speed_benchmark_v2.py
# -> paper/figures/efficiency.{pdf,png} + efficiency.json
```

### Build the paper
```bash
cd paper && pdflatex GMSWA_AAAI_final && bibtex GMSWA_AAAI_final \
         && pdflatex GMSWA_AAAI_final && pdflatex GMSWA_AAAI_final
```

---

## Deferred experiments (GPU-gated) — see `paper/TODO_experiments.md`

1. **★ Memory-branch-from-scratch** (`disable_local`) — converts the core credit-assignment
   claim from hypothesis to evidence. **Code ready:**
   ```bash
   bash flash-linear-attention/flame/run_gmswa_memonly_340M.sh
   bash flash-linear-attention/flame/run_eval_memonly.sh
   python scripts/analysis/analyze_suite.py SWA GMSWA-v5conv GMSWA-memonly GDN
   ```
2. **★ Multi-seed + confidence intervals** for all close comparisons (the statistics gate).
3. Verify citation arXiv IDs · 4. one external-hybrid head-to-head · 5. 1B replication ·
   6. short-conv ablation.

---

## Status

Paper text complete; compiles to a clean AAAI PDF; two review rounds passed
(verdict: *weak accept, text-conditional*). The two items blocking a higher verdict are
deferred experiments 1 and 2 above, both GPU-gated.
