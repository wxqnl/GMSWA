#!/usr/bin/bash
# GM-SWA v3 340M — architecture fix over v2:
#   - mem_separate_proj=true  (dedicated NoPE retrieval q/k for the memory)
#   - mix_gate_logit_bias=0.0, mem_gate_logit_bias=0.0  (memory in the loop from step 0)
#
# THROUGHPUT (go-forward default): seq_len 131072 + grad_accum 1.
#   Global batch is UNCHANGED vs v2: 131072*1*1*8 == 65536*1*2*8 == 1,048,576 tok/step.
#   It only removes one accumulation pass + uses a bigger packing buffer (better MFU).
#   NOTE: the currently-RUNNING v3 job was launched with the v2-matched recipe
#   (seq_len 65536, grad_accum 2) for a clean A/B — these new values apply to
#   future relaunches only.
#   context_len stays 2048 here (matches v2). To train the memory on longer
#   eviction ranges, raise context_len (e.g. 4096/8192) — that is the lever that
#   actually changes the model's effective context, not seq_len.
set -euo pipefail

VENV=/home/user01/Minko/GMSWA/.venv311
FLAME=/home/user01/Minko/GMSWA/flash-linear-attention/flame
RUN=$FLAME/saves/GMSWA-340M-v5conv-10k

cd "$FLAME"
mkdir -p "$RUN" "$RUN/logs"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8

"$VENV/bin/torchrun" \
  --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend c10d --rdzv_endpoint "localhost:29531" \
  --local-ranks-filter 0 --role rank --tee 3 \
  --log-dir "$RUN/logs" \
  -m flame.train \
  --job.dump_folder "$RUN" \
  --model.config configs/gated_mem_swa_v5conv_340M.json \
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
  --training.seq_len 131072 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
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
