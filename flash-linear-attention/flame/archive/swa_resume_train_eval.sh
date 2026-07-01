#!/usr/bin/bash
# After the GM-SWA eval frees the GPUs: resume SWA baseline training from its
# latest checkpoint (step-1000), then convert + eval it (fixed 8-GPU settings).
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
SW=$FLAME/saves/SWA-340M-v2-10k;  SWE=$ROOT/eval_results/SWA-340M-v2-10k
SWC=$FLAME/configs/swa_baseline_340M.json
mkdir -p "$SWE"

echo "[$(date)] waiting for GM-SWA eval to finish (free GPUs) ..."
while pgrep -f "lm_eval_fla\.py" >/dev/null; do sleep 30; done
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [[ "${U:-999999}" -lt 8000 ]] && break; sleep 30
done

echo "[$(date)] resuming SWA training (from latest checkpoint, load_step -1) ..."
env -u CUDA_VISIBLE_DEVICES bash "$FLAME/run_swa_340M_10k.sh" 2>&1 | tee -a "$SW/train.console.log"

if [[ ! -d "$SW/checkpoint/step-$STEP" ]]; then
  echo "[$(date)] ERROR: SWA step-$STEP missing after training. Aborting."; exit 1
fi
sleep 30
echo "[$(date)] converting SWA -> HF ..."
( cd "$FLAME" && "$PY" -m flame.utils.convert_dcp_to_hf \
    --path "$SW" --step "$STEP" --config "$SWC" --tokenizer "$TOK" ) 2>&1 | tee "$SWE/convert.log"

echo "[$(date)] SWA short eval (8-GPU, max_length=2048, bs=16) ..."
"${ACC[@]}" --model hf \
  --model_args "pretrained=$SW,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
  --tasks "$SHORT" --batch_size 16 --output_path "$SWE/short" 2>&1 | tee "$SWE/short.log"
echo "[$(date)] SWA long probe (non-fatal) ..."
"${ACC[@]}" --model hf \
  --model_args "pretrained=$SW,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
  --tasks "$LONG" --batch_size 1 --metadata '{"max_seq_lengths":[2048,4096,8192]}' \
  --output_path "$SWE/long" 2>&1 | tee "$SWE/long.log" || echo "[$(date)] SWA long failed (non-fatal)"

echo "[$(date)] SWA DONE. Compare eval_results/GMSWA-340M-v2-10k vs eval_results/SWA-340M-v2-10k"
