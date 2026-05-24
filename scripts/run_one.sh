#!/usr/bin/env bash
# Train + convert + evaluate ONE (model, scale) combination.
#
# Usage:
#   bash scripts/run_one.sh <model> <scale>
#
# Example:
#   bash scripts/run_one.sh gated_mem_swa 340M
#   bash scripts/run_one.sh transformer   1B
#
# Knobs (env vars, all optional):
#   NGPU                  default 8
#   SKIP_TRAIN=1          skip training, only convert+eval (use to re-eval an existing run)
#   SKIP_EVAL=1           skip evaluation
#   RESUME=1              resume training from latest checkpoint (load_step -1)
#   SHORT_TASKS           comma list of lm-eval short-context tasks
#   LONG_TASKS            comma list of lm-eval long-context tasks
#   WANDB=1               enable wandb logging (default on)
#
# Paths (override as env vars if your layout differs):
#   ROOT                  default /home/user01/Minko/GMSWA
#   TOKENIZER             default /home/user01/Minko/models/gla-tokenizer
#   DATASET               default /home/user01/Minko/datasets/fineweb_edu_100BT

set -euo pipefail

MODEL=${1:?"usage: bash run_one.sh <model> <scale>"}
SCALE=${2:?"usage: bash run_one.sh <model> <scale>"}

STEPS_OVERRIDE=${STEPS:-}
SEQ_LEN_OVERRIDE=${SEQ_LEN:-}
GRAD_ACCUM_OVERRIDE=${GRAD_ACCUM:-}
CKPT_INTERVAL_OVERRIDE=${CKPT_INTERVAL:-}
CONTEXT_LEN=${CONTEXT_LEN:-4096}

ROOT=${ROOT:-/home/user01/Minko/GMSWA}
FLA=$ROOT/flash-linear-attention
FLAME=$FLA/flame
TOKENIZER=${TOKENIZER:-/home/user01/Minko/models/gla-tokenizer}
DATASET=${DATASET:-/home/user01/Minko/datasets/fineweb_edu_100BT}
NGPU=${NGPU:-8}
WANDB=${WANDB:-1}

# Use the repository checkout so newly-added local model types are registered.
export PYTHONPATH="$FLA:${PYTHONPATH:-}"

CONFIG=$FLAME/configs/${MODEL}_${SCALE}.json
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"; exit 2
fi

RUN_NAME=${MODEL}-${SCALE}
SAVE_DIR=$FLAME/saves/${RUN_NAME}
EVAL_DIR=$ROOT/eval_results/${RUN_NAME}
mkdir -p "$EVAL_DIR"

# ---------- per-scale hyperparameters ----------
case "$SCALE" in
  340M)
    LR=7e-4;   WARMUP=5000;   LR_MIN=7e-5;  STEPS=50000;  CKPT_INTERVAL=5000
    SEQ_LEN=32768; GRAD_ACCUM=4 ;;
  1B)
    LR=1e-3;   WARMUP=10000;  LR_MIN=1e-4;  STEPS=100000; CKPT_INTERVAL=10000
    SEQ_LEN=16384; GRAD_ACCUM=8 ;;
  *) echo "ERROR: unsupported scale $SCALE (use 340M or 1B)"; exit 2 ;;
esac

STEPS=${STEPS_OVERRIDE:-$STEPS}
SEQ_LEN=${SEQ_LEN_OVERRIDE:-$SEQ_LEN}
GRAD_ACCUM=${GRAD_ACCUM_OVERRIDE:-$GRAD_ACCUM}
CKPT_INTERVAL=${CKPT_INTERVAL_OVERRIDE:-$CKPT_INTERVAL}

# tokens / step = SEQ_LEN * GRAD_ACCUM * NGPU
echo "============================================================"
echo "  RUN     : $RUN_NAME"
echo "  config  : $CONFIG"
echo "  save    : $SAVE_DIR"
echo "  eval    : $EVAL_DIR"
echo "  tokens/step = ${SEQ_LEN} x ${GRAD_ACCUM} x ${NGPU} = $(( SEQ_LEN * GRAD_ACCUM * NGPU ))"
echo "  total   = $(( SEQ_LEN * GRAD_ACCUM * NGPU * STEPS / 1000000000 ))B tokens"
echo "============================================================"

# ---------- TRAIN ----------
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  LOAD_STEP=-1   # always resume if checkpoint exists
  EXTRA_ARGS=()
  if [[ "$WANDB" == "1" ]]; then
    EXTRA_ARGS+=( --metrics.enable_wandb )
  fi
  pushd "$FLAME" >/dev/null
  NGPU=$NGPU bash train.sh \
    --job.dump_folder "$SAVE_DIR" \
    --model.config "$CONFIG" \
    --model.tokenizer_path "$TOKENIZER" \
    --optimizer.name AdamW \
    --optimizer.eps 1e-15 \
    --optimizer.lr "$LR" \
    --lr_scheduler.warmup_steps "$WARMUP" \
    --lr_scheduler.lr_min "$LR_MIN" \
    --lr_scheduler.decay_type cosine \
    --lr_scheduler.decay_ratio 0.2 \
    --training.batch_size 1 \
    --training.context_len "$CONTEXT_LEN" \
    --training.gradient_accumulation_steps "$GRAD_ACCUM" \
    --training.steps "$STEPS" \
    --training.skip_nan_inf \
    --training.seq_len "$SEQ_LEN" \
    --training.dataset "$DATASET" \
    --training.dataset_split train \
    --training.seed 0 \
    --checkpoint.interval "$CKPT_INTERVAL" \
    --metrics.log_freq 1 \
    --checkpoint.folder checkpoint \
    --training.num_workers 8 \
    --training.prefetch_factor 2 \
    --checkpoint.export_dtype bfloat16 \
    --checkpoint.enable_checkpoint \
    --checkpoint.load_step "$LOAD_STEP" \
    --training.data_parallel_shard_degree "$NGPU" \
    --activation_checkpoint.mode none \
    --training.streaming \
    --training.varlen \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$EVAL_DIR/train.log"
  popd >/dev/null
fi

# ---------- CONVERT (DCP → HF) ----------
if [[ ! -f "$SAVE_DIR/config.json" || "${FORCE_CONVERT:-0}" == "1" ]]; then
  pushd "$FLAME" >/dev/null
  python -m flame.utils.convert_dcp_to_hf \
    --path "$SAVE_DIR" \
    --step "$STEPS" \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER" \
    2>&1 | tee -a "$EVAL_DIR/convert.log"
  popd >/dev/null
fi

# ---------- EVAL ----------
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  bash "$ROOT/scripts/eval_one.sh" "$RUN_NAME" "$SAVE_DIR" "$EVAL_DIR"
fi

echo "DONE: $RUN_NAME"
