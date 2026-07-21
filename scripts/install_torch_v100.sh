#!/usr/bin/env bash
set -euo pipefail
# CUDA 13 removed Volta library/offline-compilation support. Use CUDA 12.x.
uv pip install --upgrade 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu128
