#!/usr/bin/bash
# Autonomous: after the 340M study frees GPUs -> train GMSWA-v3-1B (the recall experiment)
# -> convert -> NIAH sweep + loss-vs-position. The recall win is here.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python; export PYTHONPATH="$FLA:${PYTHONPATH:-}"; export HF_ALLOW_CODE_EVAL=1
cd "$FLAME"

# 1) wait for the 340M eval pipeline to finish (GPUs free)
until grep -q "FULL EVAL PIPELINE DONE" "$ROOT/eval_results/eval_pipeline.log" 2>/dev/null; do sleep 300; done
sleep 30
echo "[$(date)] 340M study done -> launching GMSWA-v3-1B training"

# 2) train 1B (blocking, ~13-15h)
bash run_gmswa_v3_1B.sh
echo "[$(date)] 1B training finished -> converting"
$PY -m flame.utils.convert_dcp_to_hf --path saves/GMSWA-v3-1B-10k --step 10000 \
    --config configs/gated_mem_swa_v3_1B.json --tokenizer /home/user01/Minko/models/gla-tokenizer 2>&1 | tail -2
for b in saves/GMSWA-v3-1B-10k/checkpoint/step-*; do [ "$(basename "$b")" != "step-10000" ] && rm -rf "$b"; done

# 3) eval: NIAH recall sweep (the headline) + loss-vs-position
CK=$FLAME/saves/GMSWA-v3-1B-10k
echo "[$(date)] loss-vs-position GMSWA-v3-1B"
CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/scripts/analysis/ppl_vs_position.py" GMSWA-v3-1B "$CK" 8192 120 > "$ROOT/eval_results/ppl_GMSWA-v3-1B.out" 2>&1 || true
echo "[$(date)] NIAH sweep GMSWA-v3-1B"
bash eval_suite.sh GMSWA-v3-1B=$CK
echo "[$(date)] 1B RECALL EXPERIMENT DONE"
