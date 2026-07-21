#!/usr/bin/env python3
from __future__ import annotations

import sys

import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime used by PyTorch: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("Install a CUDA-enabled PyTorch build before training.")

    cuda_major = 0
    if torch.version.cuda:
        cuda_major = int(torch.version.cuda.split(".", maxsplit=1)[0])

    saw_volta = False
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        saw_volta = saw_volta or props.major == 7
        bf16 = torch.cuda.is_bf16_supported()
        print(
            f"GPU {index}: {props.name}; compute capability "
            f"{props.major}.{props.minor}; VRAM {props.total_memory / 2**30:.1f} GiB; "
            f"BF16 supported by PyTorch: {bf16}"
        )

    if saw_volta and cuda_major >= 13:
        raise SystemExit(
            "V100/Volta detected with a CUDA 13 PyTorch wheel. CUDA 13 removed "
            "Volta library/offline-compilation support. Install the CUDA 12.8 "
            "wheel with scripts/install_torch_v100.sh."
        )

    recommended = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    print(f"Recommended --amp value: {recommended}")


if __name__ == "__main__":
    main()
