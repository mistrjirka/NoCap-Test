#!/usr/bin/env bash
set -euo pipefail
# V100 uses FP16 plus GradScaler because it has no native BF16 tensor cores.
exec torchrun --standalone --nproc_per_node=1 train.py \
  --preset baseline \
  --amp fp16 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --weight_decay 0.1 \
  --val_loss_every 128 \
  --target_val_loss 3.3821 \
  --stop_at_target \
  --output_dir runs
