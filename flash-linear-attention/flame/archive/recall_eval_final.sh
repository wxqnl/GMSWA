#!/usr/bin/bash
# Rigorous NIAH recall-vs-length sweep for ALL THREE models on the FIXED decode
# path: SWA (no memory), GMSWA-v2 (broken memory), GMSWA-v3 (architecture fix).
# Per-length invocations so --limit applies cleanly per length.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery"
LENS="512 1024 2048 4096 8192"
LIMIT=${LIMIT:-100}
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'
disk_ok(){ local a=$(df --output=avail -BG /home|tail -1|tr -dc '0-9'); [ "${a:-0}" -ge 8 ]; }
declare -A M=( [SWA]="$FLAME/saves/SWA-340M-v2-10k" [GMSWA-v2]="$FLAME/saves/GMSWA-340M-v2-10k" [GMSWA-v3]="$FLAME/saves/GMSWA-340M-v3-10k" )
for name in SWA GMSWA-v2 GMSWA-v3; do
  ck=${M[$name]}; [[ -f "$ck/model.safetensors" ]] || { echo "skip $name"; continue; }
  out=$ROOT/eval_results/niah_fixed/$name; mkdir -p "$out"
  for L in $LENS; do
    disk_ok || { echo "[$(date)] ABORT $name @$L: disk"; break; }
    echo "[$(date)] $name NIAH @ $L | limit=$LIMIT"
    "${ACC[@]}" --model hf --verbosity ERROR \
      --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
      --tasks "$TASKS" --metadata "{\"max_seq_lengths\":[$L]}" --limit "$LIMIT" --batch_size 1 \
      --output_path "$out/ruler_$L" 2>&1 | grep -avE "$NOISE" > "$out/ruler_$L.log" || echo "[$(date)] $name @$L FAILED"
  done
done
echo "[$(date)] NIAH-final DONE. disk: $(df -BG --output=avail /home|tail -1)"
