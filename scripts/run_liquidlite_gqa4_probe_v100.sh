#!/usr/bin/env bash
set -euo pipefail
steps="${1:-1536}"
if (( $# > 0 )); then shift; fi

exec bash "$(dirname "$0")/run_loss_velocity_probe_v100.sh" \
  liquidlite512-gqa4 "$steps" \
  --learning_rate 0.00155 \
  --lr_min_peak 0.00130 \
  --lr_max_peak 0.00180 \
  "$@"
