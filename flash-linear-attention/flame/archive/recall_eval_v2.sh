#!/usr/bin/bash
# Long-context / recall eval (v2, disk-safe). Both models are already trained+converted,
# so this runs immediately. Fixes vs v1:
#   - DROP ruler_vt/cwe/fwe (they spewed "reduces chain length" warnings -> 11GB log -> disk full)
#   - keep clean NIAH tasks (single/multikey/multiquery) for the recall-vs-length curve
#   - --verbosity ERROR + filter generation spew from logs
#   - disk guard: abort a phase if <8GB free
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")

RULER_TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery"
RULER_LENS='{"max_seq_lengths":[512,1024,2048,4096,8192]}'
LIMIT_RULER=${LIMIT_RULER:-50}
RECALL_TASKS="swde,fda,squad_completion"
LIMIT_RECALL=${LIMIT_RECALL:-500}
# strip the noisy synthetic-generation lines so logs stay tiny
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'

GM=$FLAME/saves/GMSWA-340M-v2-10k;  GME=$ROOT/eval_results/GMSWA-340M-v2-10k
SW=$FLAME/saves/SWA-340M-v2-10k;     SWE=$ROOT/eval_results/SWA-340M-v2-10k

disk_ok () { local a=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9'); [ "${a:-0}" -ge 8 ]; }

recall_for () {  # $1 RUN  $2 EVAL_DIR  $3 LABEL
  mkdir -p "$2"
  if ! disk_ok; then echo "[$(date)] ABORT ($3): <8GB disk free"; return 1; fi
  echo "[$(date)] ($3) RULER NIAH sweep [512..8192] | 8-GPU | limit=$LIMIT_RULER"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
    --tasks "$RULER_TASKS" --metadata "$RULER_LENS" \
    --limit "$LIMIT_RULER" --batch_size 1 \
    --output_path "$2/ruler" 2>&1 | grep -avE "$NOISE" > "$2/ruler.log" || echo "[$(date)] ($3) RULER failed"
  if ! disk_ok; then echo "[$(date)] ABORT ($3) recall: <8GB disk free"; return 1; fi
  echo "[$(date)] ($3) recall-intensive (swde,fda,squad_completion) | 8-GPU | limit=$LIMIT_RECALL"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$RECALL_TASKS" --limit "$LIMIT_RECALL" --batch_size 8 \
    --output_path "$2/recall" 2>&1 | grep -avE "$NOISE" > "$2/recall.log" || echo "[$(date)] ($3) recall failed"
}

# wait only for any stray GPU users to clear
while [[ -n "$(pgrep -f '[f]lame\.train')" ]] || [[ -n "$(pgrep -f '[l]m_eval_fla')" ]]; do sleep 20; done
echo "[$(date)] starting recall/long-context eval. disk free: $(df -BG --output=avail /home | tail -1)"

[[ -f "$GM/model.safetensors" ]] && recall_for "$GM" "$GME" "GM-SWA"
[[ -f "$SW/model.safetensors" ]] && recall_for "$SW" "$SWE" "SWA"

echo "[$(date)] DONE both models. disk free: $(df -BG --output=avail /home | tail -1)"
