#!/usr/bin/env bash
set -euo pipefail
# Official stable Ampere wheel. Keep the same PyTorch build for every compared preset.
uv pip install --upgrade 'torch==2.12.1' \
  --index-url https://download.pytorch.org/whl/cu130
