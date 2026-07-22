#!/usr/bin/env python3
"""Tiny CUDA and optional torch.compile smoke test for LiquidLite MoE routing."""

from __future__ import annotations

import argparse

import torch

from moe_model import MoEGPT, MoEGPTConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the MoE CUDA smoke test")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = MoEGPTConfig(
        vocab_size=257,
        n_layer=3,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        embedding_dim=48,
        mlp_ratio=2,
        activation="relu2",
        conv_layers=(0,),
        moe_layers=(2,),
        moe_num_routed_experts=7,
        moe_top_k=4,
        moe_expert_dim=16,
        moe_shared_expert_dim=16,
        moe_capacity_factor=1.10,
    )
    base_model = MoEGPT(config).to(device).train()
    model: torch.nn.Module = base_model
    if args.compile:
        model = torch.compile(model)

    tokens = torch.randint(0, config.vocab_size, (2, 64), device=device)
    targets = torch.randint(0, config.vocab_size, (2, 64), device=device)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        _, loss = model(tokens, targets, return_logits=False)
        assert loss is not None
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite MoE loss: {loss.item()}")
    status = base_model.moe_status()
    assert status is not None
    print(
        "CUDA MoE forward/backward OK; "
        f"GPU={torch.cuda.get_device_name(0)}; dtype={dtype}; "
        f"compile={args.compile}; objective={loss.item():.4f}; "
        f"active={status['active_experts']}/{status['total_experts']}; "
        f"drop={float(status['drop_rate']):.3%}; load_cv={float(status['load_cv']):.3f}"
    )


if __name__ == "__main__":
    main()
