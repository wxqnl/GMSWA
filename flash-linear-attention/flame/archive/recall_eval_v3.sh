#!/usr/bin/bash
# RULER length-sweep (v3) — the core experiment, fixed.
# v2 bug: a single multi-length task with --limit put all samples in the 512 bucket.
# Fix: ONE eval invocation per length, so --limit applies cleanly per length.
# Disk-safe: --verbosity ERROR, filtered logs, per-phase disk guard.
set -uo pipefail

ROOT=/home/user01/Minko/GMSWA
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")

TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery"
LENS="512 1024 2048 4096 8192"
LIMIT=${LIMIT:-100}
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'

GM=$FLAME/saves/GMSWA-340M-v2-10k;  GME=$ROOT/eval_results/GMSWA-340M-v2-10k
SW=$FLAME/saves/SWA-340M-v2-10k;     SWE=$ROOT/eval_results/SWA-340M-v2-10k

disk_ok () { local a=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9'); [ "${a:-0}" -ge 8 ]; }

sweep () {  # $1 RUN  $2 EVAL_DIR  $3 LABEL
  for L in $LENS; do
    if ! disk_ok; then echo "[$(date)] ABORT ($3 @$L): <8GB disk"; return 1; fi
    echo "[$(date)] ($3) NIAH @ length $L | 8-GPU | limit=$LIMIT"
    "${ACC[@]}" --model hf --verbosity ERROR \
      --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
      --tasks "$TASKS" --metadata "{\"max_seq_lengths\":[$L]}" \
      --limit "$LIMIT" --batch_size 1 \
      --output_path "$2/ruler_$L" 2>&1 | grep -avE "$NOISE" > "$2/ruler_$L.log" || echo "[$(date)] ($3 @$L) failed"
  done
}

while [[ -n "$(pgrep -f '[l]m_eval_fla')" ]]; do sleep 20; done
echo "[$(date)] RULER per-length sweep starting. disk: $(df -BG --output=avail /home | tail -1)"
[[ -f "$GM/model.safetensors" ]] && sweep "$GM" "$GME" "GM-SWA"
[[ -f "$SW/model.safetensors" ]] && sweep "$SW" "$SWE" "SWA"
echo "[$(date)] SWEEP DONE both models. disk: $(df -BG --output=avail /home | tail -1)"
