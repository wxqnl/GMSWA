set -euo pipefail
# GMSWA-v3 1B (1.34B) — THE recall experiment: 1B params crosses the recall-emergence threshold.
# Memory-safe: seq_len 65536 x accum 2 = same ~1M tok/step global batch as 340M.

VENV=/home/user01/Minko/GMSWA/.venv311
FLAME=/home/user01/Minko/GMSWA/flash-linear-attention/flame
RUN=$FLAME/saves/GMSWA-v3-1B-10k

cd "$FLAME"
mkdir -p "$RUN" "$RUN/logs"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

"$VENV/bin/torchrun" \
  --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend c10d --rdzv_endpoint "localhost:29526" \
  --local-ranks-filter 0 --role rank --tee 3 \
  --log-dir "$RUN/logs" \
  -m flame.train \
  --job.dump_folder "$RUN" \
  --model.config configs/gated_mem_swa_v3_1B.json \
  --model.tokenizer_path /home/user01/Minko/models/gla-tokenizer \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 5e-4 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 5e-5 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.decay_ratio 0.2 \
  --training.batch_size 1 \
  --training.context_len 2048 \
  --training.seq_len 65536 \
  --training.varlen \
  --training.gradient_accumulation_steps 2 \
  --training.steps 10000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 8 \
  --training.prefetch_factor 2 \
  --training.seed 0 \
  --training.data_parallel_shard_degree 8 \
  --activation_checkpoint.mode full \
  --checkpoint.enable_checkpoint \
  --checkpoint.folder "$RUN/checkpoint" \
  --checkpoint.interval 2000 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10 \
  --metrics.enable_tensorboard \
  --metrics.save_tb_folder tb
