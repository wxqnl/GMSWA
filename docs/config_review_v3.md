# GMSWA v3 config review

Authoritative training configs are under `flash-linear-attention/flame/configs/`. The root-level prototype `config*.json` files were removed because they described old small-scale experiments and could be mistaken for current training configs.

## Clean 1B configs

- `gated_mem_swa_v3_clean_1B.json`
  - 1.344B parameters when instantiated.
  - `model_type = gated_mem_swa`
  - 24 layers, hidden size 2048, 32 query heads, 8 KV heads.
  - `window_size = 512`, `max_position_embeddings = 32768`, `rope_theta = 1000000.0`.
  - `disable_memory = false`, `disable_local = false`.
  - **Critical:** `mem_evicted_only = true`; the memory branch only ingests tokens that have left the SWA window.
  - **Critical:** `mem_separate_proj = true`; memory retrieval uses separate NoPE projections.
  - `mem_use_short_conv = true`, `mem_conv_size = 4`.
  - Gate biases are neutral: `mem_gate_logit_bias = 0.0`, `mix_gate_logit_bias = 0.0`.

## Transformer-hybrid 1B variants

All three hybrid configs use the same full-Transformer insertion pattern:

```json
"attn": {
  "layers": [3, 7, 11, 15, 19, 23],
  "num_heads": 32,
  "num_kv_heads": 8,
  "qkv_bias": false,
  "qk_norm": false,
  "window_size": null,
  "rope_theta": 1000000.0
}
```

That means six of the twenty-four layers use full softmax Transformer attention, while the other eighteen layers use the named core mechanism.

- `gated_deltanet_transformer_hybrid_1B.json`
  - Instantiates as 18 GatedDeltaNet layers + 6 full Attention layers.
  - Parameter count: about 1.478B.

- `swa_transformer_hybrid_1B.json`
  - Instantiates as 18 memory-disabled GatedMemSWA/SWA layers + 6 full Attention layers.
  - Parameter count: about 1.213B.

- `gated_mem_swa_transformer_hybrid_1B.json`
  - Instantiates as 18 clean evicted-only GMSWA layers + 6 full Attention layers.
  - Parameter count: about 1.311B.
  - **Critical:** non-Transformer GMSWA layers keep `mem_evicted_only = true`.

## Implementation note

GatedDeltaNet already had an `attn` hybrid mechanism. v3 adds the same real mechanism to `fla/models/gated_mem_swa/` so that SWA/GMSWA hybrid configs are not silently ignored.

Validation command used before commit:

```bash
PYTHONPATH=flash-linear-attention /data/Minko/GMSWA/.venv311/bin/python - <<PY
import fla.models
from transformers import AutoConfig, AutoModelForCausalLM
for path in [...]:
    cfg = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_config(cfg)
    print(path, sum(p.numel() for p in model.parameters()))
PY
```
