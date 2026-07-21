#!/usr/bin/env bash
set -euo pipefail
preset="${1:-liquidlite512}"
steps="${2:-1024}"
exec torchrun --standalone --nproc_per_node=1 train.py \
  --preset "$preset" \
  --amp bf16 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_batch_size 16 \
  --num_iterations "$steps" \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 256 \
  --weight_decay 0.1 \
  --val_loss_every 128 \
  --no-stop_at_target \
  --output_dir runs-probe
