#!/usr/bin/env bash
set -euo pipefail
steps="${1:-1536}"
if (( $# > 0 )); then shift; fi
exec torchrun --standalone --nproc_per_node=1 train.py \
  --preset liquidlite512-moe \
  --amp fp16 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_batch_size 16 \
  --num_iterations "$steps" \
  --learning_rate 0.00145 \
  --lr_schedule loss-velocity-wsd \
  --warmup_iters 256 \
  --warmdown_iters 192 \
  --lr_search_start 256 \
  --lr_search_end 1024 \
  --lr_velocity_window 64 \
  --lr_upscale_factor 1.06 \
  --lr_downscale_factor 1.05 \
  --lr_factor_decay 0.90 \
  --lr_velocity_trigger 0.12 \
  --lr_velocity_accept 0.03 \
  --lr_velocity_ema_beta 0.80 \
  --lr_loss_rise_guard 0.06 \
  --lr_min_peak 0.00110 \
  --lr_max_peak 0.00165 \
  --weight_decay 0.1 \
  --val_loss_every 128 \
  --no-stop_at_target \
  --output_dir runs-probe "$@"
