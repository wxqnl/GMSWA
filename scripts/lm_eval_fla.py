#!/usr/bin/env python
"""lm-eval launcher that registers FLA custom architectures before running.

The HF checkpoints produced by `convert_dcp_to_hf` store only `config.json` +
weights (no remote modeling code / auto_map), so loading a custom `model_type`
such as `gated_mem_swa` requires `import fla`, which registers it with the
transformers Auto* classes. `python -m lm_eval` never imports fla, so it fails
with "model type ... not recognized". This shim imports fla first, then hands
off to the standard lm_eval CLI — use it exactly like `python -m lm_eval ...`.

Make sure the local checkout is importable, e.g.:
    PYTHONPATH=/home/user01/Minko/GMSWA/flash-linear-attention \
        python scripts/lm_eval_fla.py --model hf --model_args pretrained=... --tasks ...
"""
import fla  # noqa: F401  — registers gated_mem_swa (and other FLA archs)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
