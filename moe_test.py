#!/usr/bin/env python3
"""CPU structural tests for the LiquidLite fine-grained MoE."""

from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

from model import GPT, config_from_preset
from moe import FineGrainedMoE
from moe_model import MoEGPT, MoEGPTConfig, config_from_moe_preset


def tiny_config(*, top_k: int = 4) -> MoEGPTConfig:
    return MoEGPTConfig(
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
        moe_top_k=top_k,
        moe_expert_dim=16,
        moe_shared_expert_dim=16,
        moe_capacity_factor=1.10,
        moe_balance_loss_weight=0.01,
        moe_router_z_loss_weight=0.001,
    )


def test_parameter_counts() -> None:
    expected = {
        "liquidlite512": 111_461_888,
        "liquidlite512-moe": 111_483_392,
    }
    for preset, count in expected.items():
        model = MoEGPT(config_from_moe_preset(preset)) if preset.endswith("-moe") else GPT(config_from_preset(preset))
        assert model.parameter_count() == count, (preset, model.parameter_count())
        del model
        gc.collect()

    combined = MoEGPT(config_from_moe_preset("liquidlite512-moe", n_kv_head=4))
    assert combined.parameter_count() == 105_191_936
    del combined
    gc.collect()
    print("MoE parameter accounting OK")


def test_forward_backward_and_metrics() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = MoEGPT(config).train()
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, objective = model(tokens, targets)
    assert logits is not None and logits.shape == (2, 16, config.vocab_size)
    assert objective is not None and torch.isfinite(objective)
    # The gradient trick keeps the reported scalar equal to pure CE while the
    # router auxiliary loss still contributes gradients.
    torch.testing.assert_close(
        objective.detach().float(), model.last_cross_entropy
    )
    assert model.last_moe_aux_loss.item() > 0
    objective.backward()
    router = next(
        block.mlp.router
        for block in model.blocks
        if isinstance(block.mlp, FineGrainedMoE)
    )
    assert router.weight.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    status = model.moe_status()
    assert status is not None
    assert status["layers"] == 1
    assert status["active_experts"] == 5
    assert status["total_experts"] == 8
    assert status["top_k"] == 4
    assert 0 <= float(status["drop_rate"]) <= 1
    assert abs(sum(status["loads"]) - 1.0) < 1e-5
    print(
        "MoE forward/backward/metrics OK; "
        f"CE={model.last_cross_entropy.item():.4f}; "
        f"aux={model.last_moe_aux_loss.item():.4f}; "
        f"drop={float(status['drop_rate']):.3%}"
    )


def test_evaluation_is_pure_cross_entropy() -> None:
    config = tiny_config()
    model = MoEGPT(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 12))
    targets = torch.randint(0, config.vocab_size, (2, 12))
    with torch.no_grad():
        logits, loss = model(tokens, targets)
    assert logits is not None and loss is not None
    manual = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    torch.testing.assert_close(loss, manual)
    print("MoE validation loss remains pure token cross-entropy")


def test_top3_override() -> None:
    model = MoEGPT(tiny_config(top_k=3)).train()
    tokens = torch.randint(0, model.config.vocab_size, (1, 16))
    _, loss = model(tokens, tokens, return_logits=False)
    assert loss is not None
    status = model.moe_status()
    assert status is not None
    assert status["active_experts"] == 4
    assert status["top_k"] == 3
    print("top-3 routed override OK (1 shared + 3 routed)")


def test_capacity_overflow_has_shared_fallback() -> None:
    config = tiny_config()
    model = MoEGPT(config).train()
    moe = next(
        block.mlp for block in model.blocks if isinstance(block.mlp, FineGrainedMoE)
    )
    with torch.no_grad():
        moe.router.weight.zero_()
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(tokens, tokens)
    assert logits is not None and loss is not None
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert moe.last_drop_rate.item() > 0
    print(
        "capacity overflow remains finite through the always-active shared expert; "
        f"dropped route mass={moe.last_drop_rate.item():.1%}"
    )


def test_causality() -> None:
    torch.manual_seed(11)
    model = MoEGPT(tiny_config()).eval()
    prefix = torch.randint(0, model.config.vocab_size, (1, 9))
    suffix_a = torch.randint(0, model.config.vocab_size, (1, 7))
    suffix_b = torch.randint(0, model.config.vocab_size, (1, 7))
    x_a = torch.cat((prefix, suffix_a), dim=1)
    x_b = torch.cat((prefix, suffix_b), dim=1)
    with torch.no_grad():
        logits_a, _ = model(x_a, x_a)
        logits_b, _ = model(x_b, x_b)
    assert logits_a is not None and logits_b is not None
    torch.testing.assert_close(
        logits_a[:, : prefix.shape[1]], logits_b[:, : prefix.shape[1]]
    )
    print("MoE causal-prefix test OK")



def test_cross_sequence_independence() -> None:
    """Another sequence in the batch must not change this sequence's routes."""
    torch.manual_seed(13)
    model = MoEGPT(tiny_config()).eval()
    first = torch.randint(0, model.config.vocab_size, (1, 16))
    second_a = torch.randint(0, model.config.vocab_size, (1, 16))
    second_b = torch.randint(0, model.config.vocab_size, (1, 16))
    batch_a = torch.cat((first, second_a), dim=0)
    batch_b = torch.cat((first, second_b), dim=0)
    with torch.no_grad():
        logits_a, _ = model(batch_a, batch_a)
        logits_b, _ = model(batch_b, batch_b)
    assert logits_a is not None and logits_b is not None
    torch.testing.assert_close(logits_a[:1], logits_b[:1])
    print("MoE routing is independent across batch sequences")


def main() -> None:
    test_parameter_counts()
    test_forward_backward_and_metrics()
    test_evaluation_is_pure_cross_entropy()
    test_top3_override()
    test_capacity_overflow_has_shared_fallback()
    test_cross_sequence_independence()
    test_causality()


if __name__ == "__main__":
    main()
