#!/usr/bin/bash
# Gate-dilution diagnostic: re-eval v5conv NIAH with the SWA/memory mix forced to
# memory-only (alpha=0). If recall jumps -> the learned gate was diluting the
# memory; if not -> the memory itself can't do precise single-needle recall.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
cd "$FLAME"
LOG=$ROOT/eval_results/alpha_diag.log
CK=$FLAME/saves/GMSWA-340M-v5conv-10k
NIAH="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery"
NOISE='Generating synthetic|reduces chain|Max length|Current length|Noises|examples/s|Downloading|punkt'
ACC=("$ROOT/.venv311/bin/accelerate" launch --num_processes 8 --num_machines 1
     --main_process_port 29595 --dynamo_backend no --mixed_precision no "$ROOT/scripts/lm_eval_fla.py")

for A in 0.0 0.5; do
  out=$ROOT/eval_results/suite/GMSWA-v5conv-a${A}
  mkdir -p "$out"
  echo "[$(date)] ===== v5conv FORCE_ALPHA=$A =====" | tee -a "$LOG"
  for L in 512 2048 4096 8192; do
    echo "[$(date)] alpha=$A NIAH @ $L" | tee -a "$LOG"
    GMSWA_FORCE_ALPHA=$A "${ACC[@]}" --model hf --verbosity ERROR \
      --model_args "pretrained=$CK,dtype=bfloat16,trust_remote_code=True,max_length=8192" \
      --tasks "$NIAH" --metadata "{\"max_seq_lengths\":[$L]}" --limit 100 --batch_size 1 \
      --output_path "$out/ruler_$L" 2>&1 | grep -avE "$NOISE" > "$out/ruler_$L.log" || echo "a=$A @$L FAIL" | tee -a "$LOG"
  done
done
echo "[$(date)] ALPHA DIAG DONE" | tee -a "$LOG"
