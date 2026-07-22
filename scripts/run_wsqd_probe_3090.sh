#!/usr/bin/env bash
set -euo pipefail
preset="${1:-liquidlite512-gqa4}"
if (( $# > 0 )); then shift; fi
steps="${1:-1536}"
if (( $# > 0 )); then shift; fi
warmdown=$(( (steps + 4) / 5 ))
stable_end=$(( steps - warmdown ))
exec torchrun --standalone --nproc_per_node=1 train.py \
  --preset "$preset" --amp bf16 \
  --batch_size 16 --grad_accumulation_steps 32 --sequence_length 1024 \
  --val_batch_size 16 --num_iterations "$steps" \
  --learning_rate 0.00165 --lr_schedule wsqd --lr_wsqd_shift 512 \
  --warmup_iters 256 --warmdown_iters "$warmdown" \
  --lr_search_start 256 --lr_search_end "$stable_end" \
  --weight_decay 0.1 --val_loss_every 128 --no-stop_at_target \
  --output_dir runs-probe "$@"
