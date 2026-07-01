# v3 cleanup notes

This branch keeps the GMSWA implementation, training configs, paper sources, and reproducible eval/result summaries, while removing scratch prompts, transient backup files, generated LaTeX auxiliaries, and local run checkpoints/logs.

Important config boundary:
- `gated_mem_swa_v3_clean_1B.json` is the clean formal GMSWA 1B config: evicted-only memory, separate NoPE memory projections, and memory short-conv.
- `*_transformer_hybrid_1B.json` configs insert full Transformer attention at layers `[3, 7, 11, 15, 19, 23]`.
- SWA/GMSWA hybrid configs require the v3 model-code support in `fla/models/gated_mem_swa/`; without that patch, an `attn` field would be silently ignored.
