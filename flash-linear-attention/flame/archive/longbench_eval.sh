#!/usr/bin/bash
# LongBench eval (base-friendly EN subset) for GMSWA-v3 vs GMSWA-v2 vs SWA.
# Uses the FIXED cached-decode path. Tasks chosen for long-context sensitivity:
#   passage_retrieval_en (retrieval), 2wikimqa/hotpotqa (multi-hop), multifieldqa_en/qasper (QA),
#   triviaqa (QA w/ context). These need info beyond the 512 window -> where memory should help.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
export PYTHONPATH="$FLA:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")

TASKS="longbench_passage_retrieval_en,longbench_2wikimqa,longbench_hotpotqa,longbench_multifieldqa_en,longbench_qasper,longbench_triviaqa"
MAXLEN=${MAXLEN:-8192}
LIMIT=${LIMIT:-150}
NOISE='Generating|Downloading|Resolving|examples/s|it/s|Map:|Filter:'

# model_name  ->  checkpoint dir
declare -A M=(
  [SWA-v2]="$FLA/flame/saves/SWA-340M-v2-10k"
  [GMSWA-v2]="$FLA/flame/saves/GMSWA-340M-v2-10k"
  [GMSWA-v3]="$FLA/flame/saves/GMSWA-340M-v3-10k"
)
disk_ok(){ local a=$(df --output=avail -BG /home|tail -1|tr -dc '0-9'); [ "${a:-0}" -ge 8 ]; }

for name in SWA-v2 GMSWA-v2 GMSWA-v3; do
  ckpt=${M[$name]}
  [[ -f "$ckpt/model.safetensors" ]] || { echo "[$(date)] skip $name (no model.safetensors yet)"; continue; }
  out=$ROOT/eval_results/longbench/$name
  mkdir -p "$out"
  if ! disk_ok; then echo "[$(date)] ABORT ($name): <8GB disk"; break; fi
  echo "[$(date)] LongBench eval: $name | max_length=$MAXLEN limit=$LIMIT | 8-GPU"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$ckpt,dtype=bfloat16,trust_remote_code=True,max_length=$MAXLEN" \
    --tasks "$TASKS" --limit "$LIMIT" --batch_size 1 \
    --output_path "$out" 2>&1 | grep -avE "$NOISE" > "$out/run.log" || echo "[$(date)] ($name) FAILED"
done
echo "[$(date)] LongBench DONE. disk: $(df -BG --output=avail /home|tail -1)"
