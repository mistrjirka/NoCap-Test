#!/usr/bin/env python3
"""CPU structural tests for LiquidLite grouped-query attention."""

from __future__ import annotations

import gc

import torch

from model import GPT, GPTConfig, config_from_preset
from resume_utils import normalise_model_config


EXPECTED_PARAMETERS = {
    "liquidlite512": 111_461_888,
    "liquidlite512-gqa4": 105_170_432,
}


def test_parameter_counts_and_projection_shape() -> None:
    for preset, expected in EXPECTED_PARAMETERS.items():
        model = GPT(config_from_preset(preset))
        assert model.parameter_count() == expected, (
            preset,
            model.parameter_count(),
        )
        if preset.endswith("gqa4"):
            attention = next(
                block.mixer
                for block in model.blocks
                if block.mixer_kind == "attention"
            )
            # Q = 12*64 = 768; K and V = 4*64 = 256 each.
            assert attention.c_attn.out_features == 1_280
            assert model.config.n_kv_head == 4
            assert "GQA 12 query / 4 KV heads" in model.mixer_summary()
        del model
        gc.collect()
    print("LiquidLite/GQA parameter counts and projection shape OK")


def test_config_validation() -> None:
    try:
        GPTConfig(n_head=12, n_kv_head=5).validate()
    except ValueError as exc:
        assert "n_kv_head" in str(exc)
    else:
        raise AssertionError("A non-divisible KV-head count was accepted")
    print("GQA configuration validation OK")



def test_legacy_checkpoint_config_compatibility() -> None:
    legacy = {"n_head": 12, "n_embd": 768, "embedding_dim": 512}
    current = {
        "n_head": 12,
        "n_kv_head": 12,
        "n_embd": 768,
        "embedding_dim": 512,
    }
    assert normalise_model_config(legacy) == normalise_model_config(current)
    gqa = {**current, "n_kv_head": 4}
    assert normalise_model_config(legacy) != normalise_model_config(gqa)
    print("legacy non-GQA checkpoint config compatibility OK")


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=97,
        n_layer=3,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        embedding_dim=24,
        mlp_ratio=2,
        activation="relu2",
        conv_layers=(0,),
        conv_kernel_size=3,
    )


def test_forward_backward() -> None:
    torch.manual_seed(7)
    config = tiny_config()
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
    print(f"GQA CPU forward/backward fallback OK; loss={loss.item():.4f}")


def test_strict_causality() -> None:
    torch.manual_seed(11)
    config = tiny_config()
    model = GPT(config).eval()
    prefix = torch.randint(0, config.vocab_size, (1, 9))
    suffix_a = torch.randint(0, config.vocab_size, (1, 7))
    suffix_b = torch.randint(0, config.vocab_size, (1, 7))
    sequence_a = torch.cat((prefix, suffix_a), dim=1)
    sequence_b = torch.cat((prefix, suffix_b), dim=1)
    with torch.no_grad():
        logits_a, _ = model(sequence_a, sequence_a)
        logits_b, _ = model(sequence_b, sequence_b)
    assert logits_a is not None and logits_b is not None
    torch.testing.assert_close(
        logits_a[:, : prefix.shape[1]],
        logits_b[:, : prefix.shape[1]],
    )
    print("GQA strict causal-prefix test OK")


def main() -> None:
    test_parameter_counts_and_projection_shape()
    test_config_validation()
    test_legacy_checkpoint_config_compatibility()
    test_forward_backward()
    test_strict_causality()


if __name__ == "__main__":
    main()
