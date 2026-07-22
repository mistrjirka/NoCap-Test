#!/usr/bin/env python3
"""Fast CPU structural tests; no FineWeb dataset or CUDA is required."""

from __future__ import annotations

import gc

import torch

from model import GPT, GPTConfig, config_from_preset
from progress import ProgressDisplay


EXPECTED_COUNTS = {
    "baseline": 123_532_032,
    "dense512": 111_452_672,
    "dense512-gated": 111_845_888,
    "liquidlite512": 111_461_888,
    "liquidlite512-gqa4": 105_170_432,
    "liquidlite512-gelu": 111_461_888,
}


def check_full_parameter_count(preset: str) -> None:
    model = GPT(config_from_preset(preset))
    count = model.parameter_count()
    assert count == EXPECTED_COUNTS[preset], (preset, count)
    actual_std = model.tied_weight_std()
    expected_std = model.expected_tied_weight_std()
    assert 0.8 * expected_std <= actual_std <= 1.2 * expected_std
    print(
        f"{preset}: parameters/init OK ({count:,}; "
        f"std={actual_std:.5f}; {model.mixer_summary()})"
    )
    del model
    gc.collect()


def check_forward(
    activation: str,
    projection: str,
    embedding_dim: int,
    conv_layers: tuple[int, ...] = (),
) -> None:
    config = GPTConfig(
        vocab_size=257,
        n_layer=2,
        n_head=4,
        n_embd=64,
        embedding_dim=embedding_dim,
        mlp_ratio=2,
        activation=activation,  # type: ignore[arg-type]
        embedding_projection=projection,  # type: ignore[arg-type]
        conv_layers=conv_layers,
        conv_kernel_size=3,
    )
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(tokens, targets)
    assert logits is not None and logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"tiny {activation}/{projection}/d={embedding_dim}/conv={conv_layers}: "
        f"forward+backward OK; loss={loss.item():.4f}"
    )


def check_causality() -> None:
    torch.manual_seed(7)
    config = GPTConfig(
        vocab_size=97,
        n_layer=2,
        n_head=4,
        n_embd=32,
        embedding_dim=24,
        mlp_ratio=2,
        activation="relu2",
        conv_layers=(0,),
        conv_kernel_size=3,
    )
    model = GPT(config).eval()
    prefix = torch.randint(0, config.vocab_size, (1, 9))
    suffix_a = torch.randint(0, config.vocab_size, (1, 7))
    suffix_b = torch.randint(0, config.vocab_size, (1, 7))
    x_a = torch.cat((prefix, suffix_a), dim=1)
    x_b = torch.cat((prefix, suffix_b), dim=1)
    # Supplying targets makes forward return logits at every position.
    with torch.no_grad():
        logits_a, _ = model(x_a, x_a)
        logits_b, _ = model(x_b, x_b)
    assert logits_a is not None and logits_b is not None
    torch.testing.assert_close(logits_a[:, : prefix.shape[1]], logits_b[:, : prefix.shape[1]])
    print("liquidlite causal-prefix test OK")


def check_tty_zoom() -> None:
    display = ProgressDisplay(
        total_steps=100,
        val_every=10,
        target_val_loss=3.3821,
        warmup_iters=10,
        warmdown_iters=20,
        model_params=1,
        preset="test",
        gpu="CPU",
        amp="fp32",
        tokens_per_step=1,
    )
    low, high = display._zoom_range([4.10, 4.03, 3.98, 3.94], target=3.3821)
    assert low < 3.3821 < high
    assert high - low < 1.0
    print(f"TTY adaptive low-loss zoom OK ({low:.3f}–{high:.3f})")


def main() -> None:
    for preset in EXPECTED_COUNTS:
        check_full_parameter_count(preset)
    check_forward("gelu", "linear", 64)
    check_forward("relu2", "linear", 32)
    check_forward("relu2", "gated-silu", 32)
    check_forward("relu2", "linear", 32, conv_layers=(0,))
    check_causality()
    check_tty_zoom()


if __name__ == "__main__":
    main()
