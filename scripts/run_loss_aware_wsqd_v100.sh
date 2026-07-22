#!/usr/bin/env bash
set -euo pipefail
preset="${1:-liquidlite512-gqa4}"
if (( $# > 0 )); then shift; fi
exec torchrun --standalone --nproc_per_node=1 train.py \
  --preset "$preset" --amp fp16 \
  --batch_size 16 --grad_accumulation_steps 32 --sequence_length 1024 \
  --val_batch_size 16 --num_iterations 4768 \
  --learning_rate 0.00165 --lr_schedule loss-aware-wsqd --lr_wsqd_shift 1024 \
  --warmup_iters 256 --warmdown_iters 768 \
  --lr_search_start 256 --lr_search_end 4000 \
  --lr_velocity_window 128 --lr_velocity_trigger 0.12 \
  --lr_velocity_ema_beta 0.80 --lr_loss_rise_guard 0.06 \
  --lr_plateau_min_improvement 0.0225 --lr_plateau_drop_factor 0.80 \
  --lr_plateau_patience 2 --lr_plateau_cooldown 2 \
  --lr_plateau_min_multiplier 0.40 \
  --weight_decay 0.1 --val_loss_every 128 \
  --target_val_loss 3.3821 --stop_at_target --output_dir runs "$@"
