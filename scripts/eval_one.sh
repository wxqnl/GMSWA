#!/usr/bin/env bash
# Run the lm-eval suite on a converted HF checkpoint and save results to disk.
#
# Usage:
#   bash scripts/eval_one.sh <run_name> <model_path> <eval_dir>
#
# Tasks are split into short-context and long-context groups.
# Override task lists via env vars SHORT_TASKS / LONG_TASKS.

set -euo pipefail

RUN_NAME=${1:?"usage: eval_one.sh <run_name> <model_path> <eval_dir>"}
MODEL_PATH=${2:?"usage: eval_one.sh <run_name> <model_path> <eval_dir>"}
EVAL_DIR=${3:?"usage: eval_one.sh <run_name> <model_path> <eval_dir>"}

SHORT_TASKS=${SHORT_TASKS:-"piqa,openbookqa,hellaswag,arc_easy,arc_challenge,wikitext"}
LONG_TASKS=${LONG_TASKS:-"longbench_hotpotqa,longbench_qasper,niah_single_2"}

mkdir -p "$EVAL_DIR"

ROOT=${ROOT:-/home/user01/Minko/GMSWA}
PYTHON=${PYTHON:-python}

# Choose a single GPU for inference (340M/1B fits easily on one H100).
# To use a specific GPU set CUDA_VISIBLE_DEVICES before calling this script.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Make fla discoverable so its custom model_types register.
export PYTHONPATH="$ROOT/flash-linear-attention:${PYTHONPATH:-}"
# Custom FLA model_types (e.g. gated_mem_swa) only register on `import fla`,
# which plain `python -m lm_eval` never does — go through the launcher shim.
LM_EVAL=("$PYTHON" "$ROOT/scripts/lm_eval_fla.py")

echo "=== EVAL  $RUN_NAME ==="
echo "  short tasks: $SHORT_TASKS"
echo "  long  tasks: $LONG_TASKS"

# --- short-context tasks -----------------------------------------------------
"${LM_EVAL[@]}" \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True" \
  --tasks "$SHORT_TASKS" \
  --batch_size auto:4 \
  --output_path "$EVAL_DIR/short.json" \
  2>&1 | tee "$EVAL_DIR/short.log"

# --- long-context tasks (small batch for memory) -----------------------------
"${LM_EVAL[@]}" \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,max_length=8192" \
  --tasks "$LONG_TASKS" \
  --batch_size 1 \
  --output_path "$EVAL_DIR/long.json" \
  2>&1 | tee "$EVAL_DIR/long.log" || true   # don't abort whole run on a long-eval failure

echo "DONE: eval results at $EVAL_DIR"
