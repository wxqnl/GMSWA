#!/usr/bin/bash
# Wait for the standard-bench backfill to free the GPUs, then train v8 (gentle
# SWA-dropout p=0.15) and run its eval chain.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLAME=$ROOT/flash-linear-attention/flame
cd "$FLAME"
echo "[$(date)] v8 launcher: waiting for BACKFILL SHORT DONE..."
until grep -q "BACKFILL SHORT DONE" "$ROOT/eval_results/backfill_short.log" 2>/dev/null; do sleep 60; done
sleep 20
echo "[$(date)] backfill done -> launching v8 training"
bash run_gmswa_v8drop15_340M.sh
