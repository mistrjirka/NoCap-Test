#!/usr/bin/env bash
set -euo pipefail

# Full loss-velocity-WSD candidate. For a scheduler-neutral architecture test,
# use: bash scripts/run_probe_3090.sh liquidlite512-gqa4 1536
exec bash "$(dirname "$0")/run_loss_velocity_3090.sh" \
  liquidlite512-gqa4 \
  --learning_rate 0.00155 \
  --lr_min_peak 0.00130 \
  --lr_max_peak 0.00180 \
  "$@"
