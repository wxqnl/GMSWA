#!/usr/bin/bash
# Unified eval suite for the GMSWA paper. Runs, for each given model:
#   (1) NIAH recall-vs-length sweep [512..8192]  (in-window + extrapolation)
#   (2) recall-intensive real tasks (swde/fda/squad_completion)
# on the FIXED decode path, disk-safe. loss-vs-position is run separately (ppl_vs_position.py).
# Usage: eval_suite.sh <name1>=<ckpt_dir1> <name2>=<ckpt_dir2> ...
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention
export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")
NIAH="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery"
LENS="512 1024 2048 4096 8192"; RECALL="swde,fda,squad_completion"
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'
disk_ok(){ local a=$(df --output=avail -BG /home|tail -1|tr -dc '0-9'); [ "${a:-0}" -ge 8 ]; }
for pair in "$@"; do
  name=${pair%%=*}; ck=${pair#*=}
  [[ -f "$ck/model.safetensors" ]] || { echo "skip $name (no weights)"; continue; }
  out=$ROOT/eval_results/suite/$name; mkdir -p "$out"
  echo "[$(date)] ===== $name ====="
  for L in $LENS; do
    disk_ok || { echo "ABORT disk"; break; }
    echo "[$(date)] $name NIAH @ $L"
    "${ACC[@]}" --model hf --verbosity ERROR \
      --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
      --tasks "$NIAH" --metadata "{\"max_seq_lengths\":[$L]}" --limit 100 --batch_size 1 \
      --output_path "$out/ruler_$L" 2>&1 | grep -avE "$NOISE" > "$out/ruler_$L.log" || echo "$name @$L FAIL"
  done
  echo "[$(date)] $name recall-real"
  "${ACC[@]}" --model hf --verbosity ERROR \
    --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=2048" \
    --tasks "$RECALL" --limit 500 --batch_size 8 \
    --output_path "$out/recall" 2>&1 | grep -avE "$NOISE" > "$out/recall.log" || echo "$name recall FAIL"
done
echo "[$(date)] EVAL SUITE DONE. disk: $(df -BG --output=avail /home|tail -1)"
