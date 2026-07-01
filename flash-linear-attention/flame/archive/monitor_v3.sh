#!/usr/bin/bash
# Continuous real-time monitor for the v3 training run. Writes a live dashboard
# to STATUS every tick (tail -f it), keeps a history log, and exits only when
# training reaches step 10000 or the process dies.
FLAME=/home/user01/Minko/GMSWA/flash-linear-attention/flame
CON=$FLAME/saves/GMSWA-340M-v3-10k.train.console.log
STATUS=$FLAME/saves/GMSWA-340M-v3-10k.STATUS.txt
HIST=$FLAME/saves/GMSWA-340M-v3-10k.history.tsv
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }
# v2 baseline reference (step:loss)
v2ref(){ case $1 in 0*|1[0-9][0-9]|[1-9][0-9]) echo 10.3;; *) python3 - "$1" <<'P'
import sys
s=int(sys.argv[1]); ref={1000:3.42,2000:2.93,3000:2.80,4000:2.69,5000:2.65,6000:2.626,8000:2.592,10000:2.497}
k=min(ref,key=lambda x:abs(x-s)); print(f"{ref[k]:.3f}@{k}")
P
;; esac; }
miss=0
while true; do
  alive=$(pgrep -cf '[f]lame\.train')
  if [ "$alive" -gt 0 ]; then miss=0; else miss=$((miss+1)); fi
  sl=$(strip < "$CON" 2>/dev/null | grep -oE "step: *[0-9]+  loss: *[0-9.]+" | tail -1)
  step=$(echo "$sl" | grep -oE "step: *[0-9]+" | grep -oE "[0-9]+")
  loss=$(echo "$sl" | grep -oE "loss: *[0-9.]+" | grep -oE "[0-9.]+")
  mfu=$(strip < "$CON" 2>/dev/null | grep -oE "mfu: *[0-9.]+%" | tail -1)
  ck=$(ls $FLAME/saves/GMSWA-340M-v3-10k/checkpoint/ 2>/dev/null | tr '\n' ' ')
  disk=$(df -BG --output=avail /home | tail -1 | tr -d ' ')
  ts=$(date '+%H:%M:%S')
  step=${step:-0}
  eta="?"; [ "$step" -gt 30 ] && eta=$(python3 -c "print(f'{(10000-$step)*1.6/3600:.1f}h')" 2>/dev/null)
  ref=$(v2ref "${step:-0}" 2>/dev/null)
  {
    echo "===== GM-SWA v3 training — live @ $ts ====="
    echo "step:      ${step}/10000     ($mfu)   ETA ~${eta}"
    echo "loss:      ${loss:-?}     | v2 baseline ${ref}"
    echo "checkpoints: ${ck:-none yet}"
    echo "disk free: ${disk}   train procs: ${alive}"
    echo "fix status: decode bug FIXED (pre-eval will use corrected path)"
    [ "$miss" -ge 1 ] && echo "WARN: no train proc seen (miss=$miss/3)"
  } > "$STATUS"
  echo -e "$ts\t$step\t${loss:-NA}\t$mfu\t${disk}" >> "$HIST"
  if [ "${step:-0}" -ge 10000 ]; then echo "DONE step 10000 @ $ts" >> "$STATUS"; break; fi
  [ "$miss" -ge 3 ] && { echo "TRAINING PROC GONE @ $ts (crash or finished)" >> "$STATUS"; break; }
  sleep 90
done
echo "=== monitor exit ==="; cat "$STATUS"
echo "--- last 3 loss lines ---"; strip < "$CON" | grep -oE "step: *[0-9]+  loss: *[0-9.]+" | tail -3
strip < "$CON" | grep -iE "nan|error|traceback" | tail -3
