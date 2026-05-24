#!/usr/bin/env bash
# Recreate the exact Python environment used by GMSWA training/eval.
#
# Prerequisites on the machine:
#   - CUDA 12.x or 13.x driver
#   - Python 3.11 available (`python3.11 --version`)
#   - `uv` installed (https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
#
# What this does:
#   1. Create a fresh `.venv311` at the repo root using Python 3.11.
#   2. Install all packages at the EXACT versions captured in requirements.lock.txt.
#   3. Install local editable packages (fla, flame).
#
# Usage:
#   bash scripts/setup_env.sh           # full install
#   bash scripts/setup_env.sh --upgrade # upgrade to current pyproject pins
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCK=$ROOT/requirements.lock.txt

cd "$ROOT"

if ! command -v uv >/dev/null; then
  echo "ERROR: 'uv' not found. Install with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ ! -d .venv311 ]]; then
  echo "[setup] creating .venv311 with python3.11"
  uv venv .venv311 --python 3.11
fi
# Activate
source .venv311/bin/activate

echo "[setup] installing pinned requirements from $LOCK"
uv pip install -r "$LOCK"

echo "[setup] installing local editable: fla, flame"
uv pip install -e "$ROOT/flash-linear-attention"
uv pip install -e "$ROOT/flash-linear-attention/flame"

# Final sanity
python - <<'PY'
import torch, transformers, triton, lm_eval
import fla  # registers custom model types
print(f"  torch        {torch.__version__}     cuda available: {torch.cuda.is_available()}")
print(f"  triton       {triton.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  lm_eval      {lm_eval.__version__}")
print(f"  fla          OK (registered model_types)")
PY

echo "[setup] DONE. Activate with:  source $ROOT/.venv311/bin/activate"
