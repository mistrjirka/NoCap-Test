#!/usr/bin/env bash
set -euo pipefail
preset="${1:-dense512}"
if (( $# > 0 )); then shift; fi

# Scheduler-only ablation: adaptive peak search with the original 1,024-step decay.
exec torchrun --standalone --nproc_per_node=1 train.py \
  --preset "$preset" \
  --amp bf16 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --learning_rate 0.0018 \
  --lr_schedule loss-velocity-wsd \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --lr_search_start 256 \
  --lr_search_end 2808 \
  --lr_velocity_window 128 \
  --lr_upscale_factor 1.08 \
  --lr_downscale_factor 1.05 \
  --lr_factor_decay 0.90 \
  --lr_velocity_trigger 0.12 \
  --lr_velocity_accept 0.03 \
  --lr_velocity_ema_beta 0.80 \
  --lr_loss_rise_guard 0.06 \
  --lr_min_peak 0.00135 \
  --lr_max_peak 0.00220 \
  --weight_decay 0.1 \
  --val_loss_every 128 \
  --target_val_loss 3.3821 \
  --stop_at_target \
  --output_dir runs "$@"
