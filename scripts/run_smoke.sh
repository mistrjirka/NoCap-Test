#!/usr/bin/env bash
set -euo pipefail
python smoke_test.py
python resume_test.py
python scheduler_test.py
python gqa_test.py
python inspect_env.py
python gqa_cuda_smoke.py

torchrun --standalone --nproc_per_node=1 train.py \
  --preset dense512 \
  --amp auto \
  --no-compile \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --sequence_length 128 \
  --val_batch_size 8 \
  --validation_tokens 8192 \
  --num_iterations 10 \
  --val_loss_every 5 \
  --no-stop_at_target \
  --save_every 5 \
  --output_dir runs-smoke "$@"
