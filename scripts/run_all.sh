#!/usr/bin/env bash
# Driver: train + evaluate all (model, scale) combinations sequentially.
#
# Usage:
#   bash scripts/run_all.sh                       # everything: 6 models x 2 scales
#   bash scripts/run_all.sh --scales 340M         # only 340M
#   bash scripts/run_all.sh --models gated_mem_swa,swa --scales 1B
#
# Knobs (env vars):
#   NGPU=8                  number of GPUs (used as data_parallel_shard_degree)
#   SKIP_TRAIN=1            only convert+eval pre-existing checkpoints
#   SKIP_EVAL=1             only train, no eval
#   STOP_ON_ERROR=1         abort the whole queue on the first failure
#
# Output:
#   $ROOT/eval_results/<run_name>/{train.log,convert.log,short.json,long.json,short.log,long.log}
#   $ROOT/eval_results/summary.csv    -- one row per task per run

set -uo pipefail
ROOT=${ROOT:-/home/user01/Minko/GMSWA}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_MODELS="gated_mem_swa,swa,transformer,gated_deltanet,gsa,nsa"
DEFAULT_SCALES="340M,1B"

MODELS=$DEFAULT_MODELS
SCALES=$DEFAULT_SCALES
STOP_ON_ERROR=${STOP_ON_ERROR:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS=$2; shift 2 ;;
    --scales) SCALES=$2; shift 2 ;;
    -h|--help)
      head -n 20 "$0" | grep -E "^#"
      exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra SCALE_LIST <<< "$SCALES"

EVAL_ROOT=$ROOT/eval_results
mkdir -p "$EVAL_ROOT"

QUEUE_LOG=$EVAL_ROOT/queue.log
echo "==== run_all started $(date -Iseconds) ====" | tee -a "$QUEUE_LOG"
echo "  models: ${MODEL_LIST[*]}" | tee -a "$QUEUE_LOG"
echo "  scales: ${SCALE_LIST[*]}" | tee -a "$QUEUE_LOG"

FAILED=()
for scale in "${SCALE_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    run="${model}-${scale}"
    echo "" | tee -a "$QUEUE_LOG"
    echo "==== [$(date +%H:%M:%S)] START  $run ====" | tee -a "$QUEUE_LOG"

    if bash "$SCRIPT_DIR/run_one.sh" "$model" "$scale"; then
      echo "==== [$(date +%H:%M:%S)] DONE   $run ====" | tee -a "$QUEUE_LOG"
    else
      echo "==== [$(date +%H:%M:%S)] FAILED $run ====" | tee -a "$QUEUE_LOG"
      FAILED+=("$run")
      if [[ "$STOP_ON_ERROR" == "1" ]]; then
        echo "aborting (STOP_ON_ERROR=1)" | tee -a "$QUEUE_LOG"
        break 2
      fi
    fi
  done
done

# Aggregate results
echo "" | tee -a "$QUEUE_LOG"
echo "==== Aggregating to summary.csv ====" | tee -a "$QUEUE_LOG"
python "$SCRIPT_DIR/aggregate_eval.py" --eval-root "$EVAL_ROOT" --out "$EVAL_ROOT/summary.csv" || true

echo ""
echo "==== run_all finished $(date -Iseconds) ===="
if (( ${#FAILED[@]} )); then
  echo "FAILED RUNS:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
echo "All runs completed successfully. Summary at: $EVAL_ROOT/summary.csv"
