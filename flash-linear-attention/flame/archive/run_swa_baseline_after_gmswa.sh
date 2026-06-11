#!/usr/bin/bash
# Orchestrator: after the GM-SWA v2 eval finishes, train + eval the pure-SWA
# baseline (same config, disable_memory=true) for a direct comparison.
#
# Launch detached:
#   nohup bash run_swa_baseline_after_gmswa.sh > .../swa_pipeline.console.log 2>&1 &
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python
RUN=$FLAME/saves/SWA-340M-v2-10k
CONFIG=$FLAME/configs/swa_baseline_340M.json
TOKENIZER=/home/user01/Minko/models/gla-tokenizer
STEP=10000
EVAL_DIR=$ROOT/eval_results/SWA-340M-v2-10k
GMSWA_EVAL_LOG=$ROOT/eval_results/GMSWA-340M-v2-10k/auto_eval.console.log
LAUNCH=("$PY" "$ROOT/scripts/lm_eval_fla.py")

export PYTHONPATH="$FLA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$EVAL_DIR"
echo "[$(date)] SWA-baseline orchestrator up. Waiting for GM-SWA eval to finish ..."

# ---------- 1) wait until the GM-SWA eval pipeline has concluded ----------
# The gmswa watcher process disappears once convert+eval are done (or aborted).
while pgrep -f "eval_after_train_gmswa" >/dev/null; do sleep 60; done
if grep -q "All eval artifacts" "$GMSWA_EVAL_LOG" 2>/dev/null; then
  echo "[$(date)] GM-SWA eval completed successfully."
else
  echo "[$(date)] NOTE: GM-SWA watcher exited without a success marker (eval may have"
  echo "         aborted). Proceeding with the SWA baseline anyway so the comparison run exists."
fi

# ---------- 2) make sure the GPUs are actually free before grabbing all 8 ----------
echo "[$(date)] waiting for GPUs to be free ..."
while true; do
  [[ -z "$(pgrep -f 'flame.train')" ]] || { sleep 30; continue; }
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  if [[ "${USED:-999999}" -lt 8000 ]]; then  # <8 GB total across all cards => idle
    echo "[$(date)] GPUs idle (total used ${USED} MiB). Starting SWA training."
    break
  fi
  sleep 30
done

# ---------- 3) train pure-SWA baseline (blocking) ----------
echo "[$(date)] launching SWA baseline training ..."
bash "$FLAME/run_swa_340M_10k.sh" 2>&1 | tee "$RUN/train.console.log"
echo "[$(date)] SWA training finished."

# ---------- 4) convert DCP -> HF ----------
if [[ ! -d "$RUN/checkpoint/step-$STEP" ]]; then
  echo "[$(date)] ERROR: $RUN/checkpoint/step-$STEP missing — training did not reach the final step. Aborting."
  exit 1
fi
sleep 30
echo "[$(date)] converting SWA DCP step-$STEP -> HF ..."
pushd "$FLAME" >/dev/null
"$PY" -m flame.utils.convert_dcp_to_hf \
  --path "$RUN" --step "$STEP" --config "$CONFIG" --tokenizer "$TOKENIZER" \
  2>&1 | tee "$EVAL_DIR/convert.log"
popd >/dev/null
if ! ls "$RUN"/model*.safetensors "$RUN"/pytorch_model*.bin "$RUN"/model.safetensors.index.json >/dev/null 2>&1; then
  echo "[$(date)] ERROR: SWA conversion produced no HF weights. Aborting eval."
  exit 1
fi

# ---------- 5) eval (same task suites as GM-SWA for a fair comparison) ----------
SHORT_TASKS=${SHORT_TASKS:-"wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"}
echo "[$(date)] SWA short-context eval: $SHORT_TASKS"
"${LAUNCH[@]}" \
  --model hf \
  --model_args "pretrained=$RUN,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$SHORT_TASKS" --batch_size "auto:4" \
  --output_path "$EVAL_DIR/short" 2>&1 | tee "$EVAL_DIR/short.log"

LONG_TASKS=${LONG_TASKS:-"niah_single_1,niah_single_2"}
echo "[$(date)] SWA long-context probe: $LONG_TASKS (non-fatal)"
"${LAUNCH[@]}" \
  --model hf \
  --model_args "pretrained=$RUN,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
  --tasks "$LONG_TASKS" --batch_size 1 \
  --metadata '{"max_seq_lengths":[2048,4096,8192]}' \
  --output_path "$EVAL_DIR/long" 2>&1 | tee "$EVAL_DIR/long.log" || echo "[$(date)] long probe failed (non-fatal)."

echo "[$(date)] DONE. SWA baseline artifacts under $EVAL_DIR"
echo "[$(date)] Compare with GM-SWA at $ROOT/eval_results/GMSWA-340M-v2-10k"
