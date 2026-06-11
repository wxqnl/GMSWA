#!/usr/bin/bash
# Long-context / recall evaluation for BASE (PT-only) models — the core experiment.
# Standard short benchmarks are all <512 tokens (inside the SWA window), so GM-SWA
# and pure-SWA look identical there. This eval probes BEYOND the window, where the
# memory branch should matter:
#   (A) RULER synthetic, swept over context length [512..8192]  (base-friendly)
#   (B) recall-intensive real tasks swde/fda/squad_completion   (GSA / Based recipe)
# Runs for BOTH GM-SWA and the SWA baseline once the SWA pipeline has finished.
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python
export PYTHONPATH="$FLA:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")

# RULER: needle-in-haystack (single/multi), multi-query, variable tracking — all
# work for base models via cloze/completion. Swept across lengths spanning the
# 512 window and the 2048 training context up to 8192 (extrapolation).
RULER_TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery,ruler_vt"
RULER_LENS='{"max_seq_lengths":[512,1024,2048,4096,8192]}'
LIMIT_RULER=${LIMIT_RULER:-50}
# Real recall-intensive tasks (GSA Table 3b): IE from HTML/PDF + reading comp.
RECALL_TASKS="swde,fda,squad_completion"
LIMIT_RECALL=${LIMIT_RECALL:-500}

GM=$FLAME/saves/GMSWA-340M-v2-10k;  GME=$ROOT/eval_results/GMSWA-340M-v2-10k
SW=$FLAME/saves/SWA-340M-v2-10k;     SWE=$ROOT/eval_results/SWA-340M-v2-10k

recall_for () {  # $1 RUN  $2 EVAL_DIR  $3 LABEL
  mkdir -p "$2"
  echo "[$(date)] ($3) ===== RULER length sweep [512,1024,2048,4096,8192] | 8-GPU | limit=$LIMIT_RULER ====="
  "${ACC[@]}" --model hf \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
    --tasks "$RULER_TASKS" --metadata "$RULER_LENS" \
    --limit "$LIMIT_RULER" --batch_size 1 \
    --output_path "$2/ruler" 2>&1 | tee "$2/ruler.log" || echo "[$(date)] ($3) RULER failed"
  echo "[$(date)] ($3) ===== recall-intensive real (swde,fda,squad_completion) | 8-GPU | limit=$LIMIT_RECALL ====="
  "${ACC[@]}" --model hf \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$RECALL_TASKS" --limit "$LIMIT_RECALL" --batch_size 8 \
    --output_path "$2/recall" 2>&1 | tee "$2/recall.log" || echo "[$(date)] ($3) recall-real failed"
}

echo "[$(date)] waiting for the SWA pipeline (train+convert+short-eval) to finish ..."
while pgrep -f "[s]wa_resume_train_eval\.sh" >/dev/null; do sleep 60; done
while pgrep -f "[l]m_eval_fla\.py" >/dev/null; do sleep 30; done
while [[ -n "$(pgrep -f '[f]lame\.train')" ]]; do sleep 30; done
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [[ "${U:-999999}" -lt 8000 ]] && break; sleep 30
done
echo "[$(date)] GPUs free. Running long-context/recall eval for both models."

[[ -f "$GM/model.safetensors" ]] && recall_for "$GM" "$GME" "GM-SWA" || echo "[$(date)] GM-SWA HF missing!"
[[ -f "$SW/model.safetensors" ]] && recall_for "$SW" "$SWE" "SWA"    || echo "[$(date)] SWA HF missing!"

echo "[$(date)] LONG-CONTEXT / RECALL EVAL DONE for both models."
echo "  RULER:  $GME/ruler   vs   $SWE/ruler"
echo "  recall: $GME/recall  vs   $SWE/recall"
