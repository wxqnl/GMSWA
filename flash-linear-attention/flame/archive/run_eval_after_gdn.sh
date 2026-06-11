#!/usr/bin/bash
# Autonomous: wait for GDN training -> convert -> loss-vs-position (all models) -> eval suite -> done.
set -uo pipefail
ROOT=/home/user01/Minko/GMSWA; FLA=$ROOT/flash-linear-attention; FLAME=$FLA/flame
PY=$ROOT/.venv311/bin/python; export PYTHONPATH="$FLA:${PYTHONPATH:-}"
CON=$FLAME/saves/GDN-340M-10k.train.console.log
cd "$FLAME"

# 1) wait for GDN to reach step 10000
while true; do
  step=$(sed 's/\x1b\[[0-9;]*m//g' "$CON" 2>/dev/null | grep -oE "step: *[0-9]+" | grep -oE "[0-9]+" | tail -1)
  [ "${step:-0}" -ge 10000 ] && break
  pgrep -f "[f]lame\.train" >/dev/null || { echo "GDN train gone at ${step:-?}"; break; }
  sleep 180
done
echo "[$(date)] GDN training finished; converting"
$PY -m flame.utils.convert_dcp_to_hf --path saves/GDN-340M-10k --step 10000 \
    --config configs/gated_deltanet_340M_matched.json --tokenizer /home/user01/Minko/models/gla-tokenizer 2>&1 | tail -2
# free disk: drop GDN intermediate checkpoints
for b in saves/GDN-340M-10k/checkpoint/step-*; do [ "$(basename "$b")" != "step-10000" ] && rm -rf "$b"; done

# 2) loss-vs-position for all models (GPU 0, sequential, light)
declare -A M=(
  [SWA]=saves/SWA-340M-v2-10k [Transformer]=saves/Transformer-340M-10k
  [GMSWA-v2]=saves/GMSWA-340M-v2-10k [GMSWA-v3]=saves/GMSWA-340M-v3-10k [GDN]=saves/GDN-340M-10k )
for name in SWA Transformer GMSWA-v2 GMSWA-v3 GDN; do
  [ -f "$ROOT/eval_results/ppl_$name.out" ] && grep -q RESULT "$ROOT/eval_results/ppl_$name.out" && continue
  echo "[$(date)] loss-vs-position: $name"
  CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/scripts/analysis/ppl_vs_position.py" "$name" "$FLAME/${M[$name]}" 8192 120 \
    > "$ROOT/eval_results/ppl_$name.out" 2>&1 || echo "ppl $name failed"
done

# 3) NIAH + recall suite (8-GPU) for all models
bash eval_suite.sh \
  SWA=$FLAME/saves/SWA-340M-v2-10k \
  Transformer=$FLAME/saves/Transformer-340M-10k \
  GMSWA-v2=$FLAME/saves/GMSWA-340M-v2-10k \
  GMSWA-v3=$FLAME/saves/GMSWA-340M-v3-10k \
  GDN=$FLAME/saves/GDN-340M-10k
echo "[$(date)] FULL EVAL PIPELINE DONE"
