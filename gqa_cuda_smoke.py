#!/usr/bin/env python3
"""Tiny CUDA forward/backward check for PyTorch's native GQA path."""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also exercise torch.compile around the GQA model",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the native GQA smoke test")

    device = torch.device("cuda")
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    config = GPTConfig(
        vocab_size=257,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        embedding_dim=48,
        mlp_ratio=2,
        activation="relu2",
        conv_layers=(0,),
        conv_kernel_size=3,
    )
    model: torch.nn.Module = GPT(config).to(device).train()
    if args.compile:
        model = torch.compile(model)

    tokens = torch.randint(0, config.vocab_size, (2, 64), device=device)
    targets = torch.randint(0, config.vocab_size, (2, 64), device=device)
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        _, loss = model(tokens, targets, return_logits=False)
        assert loss is not None
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite native GQA loss: {loss.item()}")
    print(
        "native CUDA GQA forward/backward OK; "
        f"GPU={torch.cuda.get_device_name(0)}; dtype={amp_dtype}; "
        f"compile={args.compile}; loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
