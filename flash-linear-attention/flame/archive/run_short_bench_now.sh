#!/usr/bin/bash
# Standard zero-shot LM benchmark for Transformer, GMSWA-v3, GDN — runs NOW on
# GPUs 0-3 (freed when vLLM exited), in PARALLEL with the NIAH re-run on 4-7.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$FLAME"
LOG=$ROOT/eval_results/short_bench.log
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 4 --num_machines 1
     --main_process_port 29577 --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")
SHORT="wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,sciq,copa"
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'

echo "[$(date)] ===== standard short-bench START on GPUs 0-3 (Transformer, GMSWA-v3, GDN) =====" | tee -a "$LOG"
declare -A CK=(
  [Transformer-340M-10k]="$FLAME/saves/Transformer-340M-10k"
  [GMSWA-340M-v3-10k]="$FLAME/saves/GMSWA-340M-v3-10k"
  [GDN-340M-10k]="$FLAME/saves/GDN-340M-10k"
)
for name in Transformer-340M-10k GMSWA-340M-v3-10k GDN-340M-10k; do
  ck=${CK[$name]}
  [[ -f "$ck/model.safetensors" ]] || { echo "skip $name (no weights)" | tee -a "$LOG"; continue; }
  out=$ROOT/eval_results/$name; mkdir -p "$out"
  echo "[$(date)] $name short-context | tasks=$SHORT" | tee -a "$LOG"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$SHORT" --batch_size 16 \
    --output_path "$out/short" 2>&1 | grep -avE "$NOISE" > "$out/short.log" \
    || echo "$name short FAIL" | tee -a "$LOG"
done
echo "[$(date)] SHORT BENCH REST DONE" | tee -a "$LOG"
