#!/usr/bin/bash
# Re-run BOTH evals (GM-SWA + SWA baseline) with the fixed settings, after the
# current comparison driver + SWA training have finished. 8-GPU data-parallel,
# max_length=2048 (avoids the 32768-token wikitext GEMM crash), fixed batch.
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python
TOK=/home/user01/Minko/models/gla-tokenizer
STEP=10000
export PYTHONPATH="$FLA:${PYTHONPATH:-}"
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")
SHORT="wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"
LONG="niah_single_1,niah_single_2"

GM=$FLAME/saves/GMSWA-340M-v2-10k;  GME=$ROOT/eval_results/GMSWA-340M-v2-10k
SW=$FLAME/saves/SWA-340M-v2-10k;     SWE=$ROOT/eval_results/SWA-340M-v2-10k
GMC=$FLAME/configs/gated_mem_swa_340M.json
SWC=$FLAME/configs/swa_baseline_340M.json
mkdir -p "$GME" "$SWE"

have () { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }
ensure_hf () {  # RUN CONFIG EVALDIR
  have "$1" && return 0
  echo "[$(date)] converting $1 -> HF"
  ( cd "$FLAME" && "$PY" -m flame.utils.convert_dcp_to_hf \
      --path "$1" --step "$STEP" --config "$2" --tokenizer "$TOK" ) 2>&1 | tee "$3/convert.log"
}
do_eval () {  # RUN EVALDIR LABEL
  echo "[$(date)] ($3) short-context | 8-GPU | max_length=2048 | bs=16"
  "${ACC[@]}" --model hf \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$SHORT" --batch_size 16 --output_path "$2/short" 2>&1 | tee "$2/short.log"
  echo "[$(date)] ($3) long-context probe (non-fatal)"
  "${ACC[@]}" --model hf \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
    --tasks "$LONG" --batch_size 1 --metadata '{"max_seq_lengths":[2048,4096,8192]}' \
    --output_path "$2/long" 2>&1 | tee "$2/long.log" || echo "[$(date)] ($3) long failed (non-fatal)"
}

echo "[$(date)] waiting for comparison driver to exit ..."
while pgrep -f "run_comparison_evals\.sh" >/dev/null; do sleep 60; done
echo "[$(date)] waiting for SWA training to finish ..."
while [[ -n "$(pgrep -f flame.train)" ]]; do sleep 30; done
echo "[$(date)] waiting for GPUs to free ..."
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [[ "${U:-999999}" -lt 8000 ]] && break; sleep 30
done
echo "[$(date)] GPUs idle. Running both evals (fixed)."

ensure_hf "$GM" "$GMC" "$GME"; if have "$GM"; then do_eval "$GM" "$GME" "GM-SWA"; else echo "[$(date)] GM-SWA HF missing!"; fi
ensure_hf "$SW" "$SWC" "$SWE"; if have "$SW"; then do_eval "$SW" "$SWE" "SWA";    else echo "[$(date)] SWA HF missing!"; fi

echo "[$(date)] BOTH EVALS DONE."
echo "  GM-SWA: $GME/short    SWA: $SWE/short"
