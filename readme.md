# GMSWA — Gated-Memory Sliding-Window Attention

This branch is the cleaned v3 training branch for scaling GMSWA and its baselines to 1B.

GMSWA combines exact sliding-window attention with a gated-delta recurrent memory. The
formal v3 setup uses the complementary design: SWA owns the active local window and the
recurrent memory ingests only tokens that have been evicted from that window.

## What is authoritative

- Implementation: `flash-linear-attention/fla/layers/gated_mem_swa.py` and
  `flash-linear-attention/fla/models/gated_mem_swa/`.
- Training configs: `flash-linear-attention/flame/configs/`.
- Launch scripts: `flash-linear-attention/flame/run_*_1B.sh`.
- Config review: `docs/config_review_v3.md`.
- Cleanup notes: `docs/v3_cleanup.md`.

Old root-level prototype configs and scratch prompts were removed so new training jobs do
not accidentally pick up obsolete small-model settings.

## 1B configs to use

### Clean GMSWA

- `flash-linear-attention/flame/configs/gated_mem_swa_v3_clean_1B.json`
- Launch: `flash-linear-attention/flame/run_gmswa_v3_clean_1B.sh`

Critical flags:

```json
{
  "model_type": "gated_mem_swa",
  "hidden_size": 2048,
  "num_hidden_layers": 24,
  "num_heads": 32,
  "num_kv_heads": 8,
  "window_size": 512,
  "disable_memory": false,
  "mem_evicted_only": true,
  "mem_separate_proj": true,
  "mem_use_short_conv": true
}
```

### Transformer-hybrid variants

Each hybrid uses full Transformer attention at layers `[3, 7, 11, 15, 19, 23]` and
keeps the named mechanism in all other layers.

- GDN hybrid: `flash-linear-attention/flame/configs/gated_deltanet_transformer_hybrid_1B.json`
  - Launch: `flash-linear-attention/flame/run_gdn_transformer_hybrid_1B.sh`
- SWA hybrid: `flash-linear-attention/flame/configs/swa_transformer_hybrid_1B.json`
  - Launch: `flash-linear-attention/flame/run_swa_transformer_hybrid_1B.sh`
- GMSWA hybrid: `flash-linear-attention/flame/configs/gated_mem_swa_transformer_hybrid_1B.json`
  - Launch: `flash-linear-attention/flame/run_gmswa_transformer_hybrid_1B.sh`

The SWA/GMSWA hybrid configs require the v3 model-code patch in
`fla/models/gated_mem_swa/`; otherwise an `attn` field would be ignored.

## Running on a new server

Set paths by environment variables rather than editing scripts:

```bash
export GMSWA_ROOT=/data/Minko/GMSWA
export DATASET=/shared/Minko/datasets/fineweb_edu_100BT
export TOKENIZER=/data/Minko/models/gla-tokenizer
export GPUS=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export DP_SHARD_DEGREE=8

cd $GMSWA_ROOT/flash-linear-attention/flame
bash run_gmswa_v3_clean_1B.sh
```

For H100 multi-GPU jobs on the current cluster, keep `NCCL_NVLS_ENABLE=0` unless the
new server has already verified stable NVLS collectives.

## Validation already performed

The four 1B configs were loaded with `AutoConfig` and instantiated with
`AutoModelForCausalLM.from_config` under the project environment. Results:

- clean GMSWA: 24 `GatedMemSWA` layers, about 1.344B parameters.
- GMSWA/Transformer hybrid: 18 `GatedMemSWA` + 6 full `Attention` layers, about 1.311B parameters.
- SWA/Transformer hybrid: 18 memory-disabled GMSWA/SWA + 6 full `Attention` layers, about 1.213B parameters.
- GDN/Transformer hybrid: 18 `GatedDeltaNet` + 6 full `Attention` layers, about 1.478B parameters.

## Paper and results

Paper drafts, figures, and structured evaluation outputs are kept under `paper/`,
`shiyan/`, and `eval_results/`. Transient logs, scratch prompts, backup files, LaTeX
auxiliaries, checkpoints, and tensorboard runs are ignored or removed from the branch.
