#!/usr/bin/bash
# Clean GMSWA-Base rerun: complementary evicted-only memory.
# Difference from v5conv: mem_evicted_only=true, so SWA owns the active window
# and the recurrent memory receives only tokens that have left the window.
set -euo pipefail

VENV=/data/Minko/GMSWA/.venv311
FLAME=/data/Minko/GMSWA/flash-linear-attention/flame
RUN=$FLAME/saves/GMSWA-340M-v5evict-10k

cd "$FLAME"
mkdir -p "$RUN" "$RUN/logs"

export CUDA_VISIBLE_DEVICES=7
export CUDA_HOME="$VENV/cudahome"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export NCCL_NVLS_ENABLE=0

"$VENV/bin/torchrun" \
  --nnodes=1 --nproc_per_node=1 \
  --rdzv_backend c10d --rdzv_endpoint "localhost:29571" \
  --local-ranks-filter 0 --role rank --tee 3 \
  --log-dir "$RUN/logs" \
  -m flame.train \
  --job.dump_folder "$RUN" \
  --model.config configs/gated_mem_swa_v5evict_340M.json \
  --model.tokenizer_path /data/Minko/models/gla-tokenizer \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 5e-4 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 5e-5 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.decay_ratio 0.2 \
  --training.batch_size 1 \
  --training.context_len 2048 \
  --training.seq_len 131072 \
  --training.varlen \
  --training.gradient_accumulation_steps 8 \
  --training.steps 10000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /shared/Minko/datasets/fineweb_edu_100BT \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 1 \
  --training.prefetch_factor 2 \
  --training.seed 0 \
  --training.data_parallel_shard_degree 1 \
  --activation_checkpoint.mode full \
  --checkpoint.enable_checkpoint \
  --checkpoint.folder "$RUN/checkpoint" \
  --checkpoint.interval 2000 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10 \
  --metrics.enable_tensorboard \
  --metrics.save_tb_folder tb
