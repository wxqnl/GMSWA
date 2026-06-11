#!/usr/bin/bash
# Auto-launch the 1B GMSWA recall experiment (paper headline) once ALL 8 GPUs
# are free again: i.e. (1) the 340M 4-GPU re-run finished AND (2) the external
# longluxi vLLM on GPUs 0-3 has exited. Does NOT touch the vLLM — just waits.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python; export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
LOG=$ROOT/eval_results/run_1B.log
cd "$FLAME"

# 1) wait for the 340M 4-GPU re-run to finish (frees GPUs 4-7)
echo "[$(date)] waiting for 340M re-run to finish..." | tee -a "$LOG"
until grep -q "4GPU REST EVAL DONE" "$ROOT/eval_results/eval_4gpu.log" 2>/dev/null; do sleep 120; done

# 2) wait for ALL 8 GPUs to be free (<2GB used) — i.e. vLLM on 0-3 gone too
echo "[$(date)] 340M done. waiting for all 8 GPUs free (vLLM to exit)..." | tee -a "$LOG"
while :; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '$1+0>2000{c++} END{print c+0}')
  [ "${busy:-9}" -eq 0 ] && break
  sleep 300
done
sleep 30
echo "[$(date)] all GPUs free -> launching 8-GPU GMSWA-v3-1B training" | tee -a "$LOG"

# 3) train 1B (blocking, ~13-15h)
bash run_gmswa_v3_1B.sh 2>&1 | tee -a "$LOG"
echo "[$(date)] 1B training finished -> converting" | tee -a "$LOG"
$PY -m flame.utils.convert_dcp_to_hf --path saves/GMSWA-v3-1B-10k --step 10000 \
    --config configs/gated_mem_swa_v3_1B.json --tokenizer /home/user01/Minko/models/gla-tokenizer 2>&1 | tail -2 | tee -a "$LOG"
for b in saves/GMSWA-v3-1B-10k/checkpoint/step-*; do [ "$(basename "$b")" != "step-10000" ] && rm -rf "$b"; done

# 4) eval: NIAH recall sweep (headline) + loss-vs-position
CK=$FLAME/saves/GMSWA-v3-1B-10k
echo "[$(date)] loss-vs-position GMSWA-v3-1B" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/scripts/analysis/ppl_vs_position.py" GMSWA-v3-1B "$CK" 8192 120 > "$ROOT/eval_results/ppl_GMSWA-v3-1B.out" 2>&1 || true
echo "[$(date)] NIAH sweep GMSWA-v3-1B" | tee -a "$LOG"
bash eval_suite.sh GMSWA-v3-1B=$CK 2>&1 | tee -a "$LOG"
echo "[$(date)] 1B RECALL EXPERIMENT DONE" | tee -a "$LOG"
