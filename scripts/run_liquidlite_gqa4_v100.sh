#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/run_loss_velocity_v100.sh" \
  liquidlite512-gqa4 \
  --learning_rate 0.00155 \
  --lr_min_peak 0.00130 \
  --lr_max_peak 0.00180 \
  "$@"
