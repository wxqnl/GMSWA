#!/usr/bin/bash
# Auto-eval for the GM-SWA v2 340M 10k-step run.
#
# Waits for the final checkpoint + training exit, converts DCP -> HF, then runs
# the lm-eval short-context zero-shot suite (+ a light long-context probe).
# Launch detached so it survives the shell:
#   nohup bash eval_after_train_gmswa.sh > .../auto_eval.console.log 2>&1 &
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python
RUN=$FLAME/saves/GMSWA-340M-v2-10k
CONFIG=$FLAME/configs/gated_mem_swa_340M.json
TOKENIZER=/home/user01/Minko/models/gla-tokenizer
STEP=${STEP:-10000}
EVAL_DIR=$ROOT/eval_results/GMSWA-340M-v2-10k
LAUNCH=("$PY" "$ROOT/scripts/lm_eval_fla.py")

export PYTHONPATH="$FLA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$EVAL_DIR"
echo "[$(date)] watcher up; waiting for $RUN/checkpoint/step-$STEP and training to exit ..."

# ---------- 1) wait for final checkpoint + training process exit ----------
while true; do
  if [[ -d "$RUN/checkpoint/step-$STEP" ]]; then
    if ! pgrep -f "flame.train" >/dev/null; then
      echo "[$(date)] final checkpoint present and training exited."
      break
    fi
  elif ! pgrep -f "flame.train" >/dev/null; then
    echo "[$(date)] WARNING: training process is gone but step-$STEP is missing."
    echo "           Latest checkpoints under $RUN/checkpoint:"
    ls -1 "$RUN/checkpoint" 2>/dev/null || echo "           (none)"
    echo "           Training likely crashed or was stopped early. Aborting auto-eval."
    echo "           To eval an earlier step: STEP=<n> bash $0"
    exit 1
  fi
  sleep 60
done
sleep 30  # let async checkpoint staging flush

# ---------- 2) convert DCP -> HF ----------
echo "[$(date)] converting DCP step-$STEP -> HF into $RUN ..."
pushd "$FLAME" >/dev/null
"$PY" -m flame.utils.convert_dcp_to_hf \
  --path "$RUN" --step "$STEP" --config "$CONFIG" --tokenizer "$TOKENIZER" \
  2>&1 | tee "$EVAL_DIR/convert.log"
popd >/dev/null

if ! ls "$RUN"/model*.safetensors "$RUN"/pytorch_model*.bin "$RUN"/model.safetensors.index.json >/dev/null 2>&1; then
  echo "[$(date)] ERROR: conversion produced no HF weights in $RUN. Aborting eval."
  exit 1
fi
echo "[$(date)] HF checkpoint ready at $RUN"

# ---------- 3) short-context zero-shot suite ----------
SHORT_TASKS=${SHORT_TASKS:-"wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"}
echo "[$(date)] short-context eval: $SHORT_TASKS"
"${LAUNCH[@]}" \
  --model hf \
  --model_args "pretrained=$RUN,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$SHORT_TASKS" \
  --batch_size "auto:4" \
  --output_path "$EVAL_DIR/short" \
  2>&1 | tee "$EVAL_DIR/short.log"

# ---------- 4) light long-context probe (non-fatal) ----------
LONG_TASKS=${LONG_TASKS:-"niah_single_1,niah_single_2"}
echo "[$(date)] long-context probe: $LONG_TASKS (non-fatal)"
"${LAUNCH[@]}" \
  --model hf \
  --model_args "pretrained=$RUN,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
  --tasks "$LONG_TASKS" \
  --batch_size 1 \
  --metadata '{"max_seq_lengths":[2048,4096,8192]}' \
  --output_path "$EVAL_DIR/long" \
  2>&1 | tee "$EVAL_DIR/long.log" || echo "[$(date)] long-context probe failed (non-fatal)."

echo "[$(date)] DONE. All eval artifacts under $EVAL_DIR"
