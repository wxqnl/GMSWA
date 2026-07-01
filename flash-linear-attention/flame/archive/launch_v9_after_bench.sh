#!/usr/bin/bash
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLAME=$ROOT/flash-linear-attention/flame
cd "$FLAME"
echo "[$(date)] v9 launcher: waiting for speed benchmark to finish..."
until grep -q "RESULT_JSON" "$ROOT/eval_results/speed_benchmark.log" 2>/dev/null; do sleep 30; done
sleep 15
echo "[$(date)] benchmark done -> launching v9 curriculum training"
bash run_gmswa_v9curr_340M.sh
