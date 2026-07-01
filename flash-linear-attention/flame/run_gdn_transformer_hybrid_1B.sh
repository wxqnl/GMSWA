#!/usr/bin/bash
# 1B launch template for GDN-TransformerHybrid-1B-10k.
# Override GMSWA_ROOT, DATASET, TOKENIZER, GPUS, NPROC_PER_NODE, DP_SHARD_DEGREE, SEQ_LEN, GRAD_ACCUM, STEPS as needed.
set -euo pipefail

GMSWA_ROOT=${GMSWA_ROOT:-/data/Minko/GMSWA}
VENV=${VENV:-$GMSWA_ROOT/.venv311}
FLAME=${FLAME:-$GMSWA_ROOT/flash-linear-attention/flame}
RUN=${RUN:-$FLAME/saves/GDN-TransformerHybrid-1B-10k}
DATASET=${DATASET:-/shared/Minko/datasets/fineweb_edu_100BT}
TOKENIZER=${TOKENIZER:-/data/Minko/models/gla-tokenizer}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DP_SHARD_DEGREE=${DP_SHARD_DEGREE:-$NPROC_PER_NODE}
SEQ_LEN=${SEQ_LEN:-65536}
GRAD_ACCUM=${GRAD_ACCUM:-2}
STEPS=${STEPS:-10000}
RDZV_PORT=${RDZV_PORT:-29534}

cd "$FLAME"
mkdir -p "$RUN" "$RUN/logs"

export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_HOME="${CUDA_HOME:-$VENV/cudahome}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

"$VENV/bin/torchrun" \
  --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv_backend c10d --rdzv_endpoint "localhost:$RDZV_PORT" \
  --local-ranks-filter 0 --role rank --tee 3 \
  --log-dir "$RUN/logs" \
  -m flame.train \
  --job.dump_folder "$RUN" \
  --model.config configs/gated_deltanet_transformer_hybrid_1B.json \
  --model.tokenizer_path "$TOKENIZER" \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 5e-4 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 5e-5 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.decay_ratio 0.2 \
  --training.batch_size 1 \
  --training.context_len 2048 \
  --training.seq_len "$SEQ_LEN" \
  --training.varlen \
  --training.gradient_accumulation_steps "$GRAD_ACCUM" \
  --training.steps "$STEPS" \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset "$DATASET" \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 8 \
  --training.prefetch_factor 2 \
  --training.seed 0 \
  --training.data_parallel_shard_degree "$DP_SHARD_DEGREE" \
  --activation_checkpoint.mode full \
  --checkpoint.enable_checkpoint \
  --checkpoint.folder "$RUN/checkpoint" \
  --checkpoint.interval 2000 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10 \
  --metrics.enable_tensorboard \
  --metrics.save_tb_folder tb
