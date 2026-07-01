#!/usr/bin/bash
# After v7 finishes (GPUs free), backfill the standard zero-shot ("short") bench
# for the 3 variants that lack it: v5conv, v6onorm, v7drop. Robust: mkdir first.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
cd "$FLAME"
LOG=$ROOT/eval_results/backfill_short.log
SHORT="wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --main_process_port 29596 --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")

echo "[$(date)] waiting for v7 eval to finish..." | tee -a "$LOG"
until grep -q "V7DROP EVAL DONE" "$ROOT/eval_results/eval_v7drop.log" 2>/dev/null; do sleep 120; done
sleep 30
echo "[$(date)] ===== backfill standard short-bench =====" | tee -a "$LOG"
for name in GMSWA-340M-v5conv-10k GMSWA-340M-v6onorm-10k GMSWA-340M-v7drop-10k; do
  ck=$FLAME/saves/$name
  [[ -f "$ck/model.safetensors" ]] || { echo "skip $name (no weights)" | tee -a "$LOG"; continue; }
  out=$ROOT/eval_results/$name; mkdir -p "$out/short"
  echo "[$(date)] $name short-bench" | tee -a "$LOG"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$SHORT" --batch_size 16 --output_path "$out/short" 2>&1 \
    | grep -avE "$NOISE" > "$out/short.log" || echo "$name short FAIL" | tee -a "$LOG"
done
echo "[$(date)] BACKFILL SHORT DONE" | tee -a "$LOG"
