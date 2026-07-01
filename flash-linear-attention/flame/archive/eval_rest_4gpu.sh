#!/usr/bin/bash
# Re-run NIAH sweep + recall for the 3 models that failed under external GPU
# contention (vLLM grabbed GPUs 0-3). Constrained to the FREE GPUs 4-7.
# SWA + Transformer already completed cleanly on 8 GPUs -> not re-run.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd "$FLAME"
LOG=$ROOT/eval_results/eval_4gpu.log
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 4 --num_machines 1
     --main_process_port 29555 --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")
NIAH="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery"
LENS="512 1024 2048 4096 8192"; RECALL="swde,fda,squad_completion"
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'
disk_ok(){ local a=$(df --output=avail -BG /home|tail -1|tr -dc '0-9'); [ "${a:-0}" -ge 8 ]; }

declare -A CK=(
  [GMSWA-v2]="$FLAME/saves/GMSWA-340M-v2-10k"
  [GMSWA-v3]="$FLAME/saves/GMSWA-340M-v3-10k"
  [GDN]="$FLAME/saves/GDN-340M-10k"
)
echo "[$(date)] ===== 4-GPU (4,5,6,7) re-run START =====" | tee -a "$LOG"
for name in GMSWA-v2 GMSWA-v3 GDN; do
  ck=${CK[$name]}
  [[ -f "$ck/model.safetensors" ]] || { echo "skip $name (no weights)" | tee -a "$LOG"; continue; }
  out=$ROOT/eval_results/suite/$name
  rm -rf "$out"; mkdir -p "$out"          # clean any failed-run remnants
  echo "[$(date)] ===== $name =====" | tee -a "$LOG"
  for L in $LENS; do
    disk_ok || { echo "ABORT disk" | tee -a "$LOG"; break; }
    echo "[$(date)] $name NIAH @ $L" | tee -a "$LOG"
    "${ACC[@]}" --model hf --verbosity ERROR \
      --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
      --tasks "$NIAH" --metadata "{\"max_seq_lengths\":[$L]}" --limit 100 --batch_size 1 \
      --output_path "$out/ruler_$L" 2>&1 | grep -avE "$NOISE" > "$out/ruler_$L.log" \
      || echo "$name @$L FAIL" | tee -a "$LOG"
  done
  echo "[$(date)] $name recall-real" | tee -a "$LOG"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$RECALL" --limit 500 --batch_size 8 \
    --output_path "$out/recall" 2>&1 | grep -avE "$NOISE" > "$out/recall.log" \
    || echo "$name recall FAIL" | tee -a "$LOG"
done
echo "[$(date)] 4GPU REST EVAL DONE. disk: $(df -BG --output=avail /home|tail -1)" | tee -a "$LOG"
