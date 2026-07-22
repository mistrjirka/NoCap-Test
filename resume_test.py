#!/usr/bin/env python3
"""CPU-only tests for exact-resume bookkeeping."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from data_loader import HEADER_INTS, MAGIC, DistributedDataLoader
from model import GPT, GPTConfig
from resume_utils import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_link_or_copy,
    atomic_torch_save,
    capture_rng_state,
    load_loader_state,
    loader_state_dict,
    restore_rng_state,
    validate_resume_checkpoint,
)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype(np.uint16).tobytes())


def test_loader_and_prefetched_batch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        shard = Path(directory) / "train.bin"
        write_shard(shard, np.arange(128, dtype=np.uint16))
        loader = DistributedDataLoader(str(shard), 2, 4, 0, 1, torch.device("cpu"))
        first_x, _ = loader.next_batch()
        prefetched_x, prefetched_y = loader.next_batch()
        saved_loader_state = loader_state_dict(loader)
        expected_x, expected_y = loader.next_batch()
        restored = DistributedDataLoader(str(shard), 2, 4, 0, 1, torch.device("cpu"))
        load_loader_state(restored, saved_loader_state)
        resumed_x, resumed_y = restored.next_batch()
        assert not torch.equal(first_x, prefetched_x)
        torch.testing.assert_close(prefetched_y[:, :-1], prefetched_x[:, 1:])
        torch.testing.assert_close(resumed_x, expected_x)
        torch.testing.assert_close(resumed_y, expected_y)
        print("data-loader state + prefetched-batch continuity OK")


def test_rng_round_trip() -> None:
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = torch.rand(8)
    _ = torch.rand(8)
    restore_rng_state(state)
    torch.testing.assert_close(torch.rand(8), expected)
    print("RNG round-trip OK")


def test_atomic_save_and_alias() -> None:
    with tempfile.TemporaryDirectory() as directory:
        latest = Path(directory) / "latest.pt"
        final = Path(directory) / "final.pt"
        atomic_torch_save(
            {"format_version": CHECKPOINT_FORMAT_VERSION, "value": 7}, latest
        )
        atomic_link_or_copy(latest, final)
        assert torch.load(final, weights_only=False)["value"] == 7
        assert not latest.with_name(latest.name + ".tmp").exists()
        assert not final.with_name(final.name + ".tmp").exists()
        print("atomic checkpoint write + alias OK")


def test_strict_compatibility() -> None:
    values = {
        "input_bin": "train/*.bin",
        "input_val_bin": "val/*.bin",
        "preset": "dense512",
        "embedding_dim": None,
        "n_kv_head": None,
        "activation": None,
        "embedding_projection": None,
        "qk_norm": None,
        "conv_layers": None,
        "conv_kernel_size": None,
        "batch_size": 16,
        "grad_accumulation_steps": 32,
        "sequence_length": 1024,
        "context_schedule": [],
        "num_iterations": 4768,
        "learning_rate": 0.0018,
        "warmup_iters": 256,
        "warmdown_iters": 1024,
        "weight_decay": 0.1,
        "val_loss_every": 128,
        "val_batch_size": 16,
        "validation_tokens": 1_048_576,
        "target_val_loss": 3.3821,
        "stop_at_target": True,
        "compile": True,
        "seed": 1337,
    }
    args = argparse.Namespace(**values)
    config = {
        "embedding_dim": 512,
        "n_head": 12,
        "n_kv_head": 12,
        "conv_layers": [],
    }
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "args": dict(values),
        "model_config": dict(config),
        "next_step": 512,
        "runtime": {
            "amp": "bf16",
            "world_size": 1,
            "gpu": "RTX 3090",
            "torch_version": torch.__version__,
        },
        "model": {},
        "optimizer": {},
        "scaler": {},
        "train_loader": {},
        "prefetched_batch": {},
        "rng_state": {},
        "training_time_ms": 1.0,
        "run_id": "test",
    }
    validate_resume_checkpoint(
        checkpoint,
        args,
        model_config=config,
        amp_name="bf16",
        world_size=1,
        gpu_name="RTX 3090",
        torch_version=torch.__version__,
    )
    changed = argparse.Namespace(**{**values, "learning_rate": 0.0017})
    try:
        validate_resume_checkpoint(
            checkpoint,
            changed,
            model_config=config,
            amp_name="bf16",
            world_size=1,
            gpu_name="RTX 3090",
            torch_version=torch.__version__,
        )
    except ValueError as exc:
        assert "learning_rate" in str(exc)
    else:
        raise AssertionError("A changed learning rate was not rejected")
    print("strict resume compatibility checks OK")


def test_model_optimizer_continuation() -> None:
    torch.manual_seed(99)
    config = GPTConfig(
        vocab_size=31,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        embedding_dim=12,
        mlp_ratio=1,
        activation="relu2",
        conv_layers=(0,),
    )
    reference = GPT(config)
    interrupted = GPT(config)
    interrupted.load_state_dict(reference.state_dict())
    reference_opt = reference.configure_optimizer(
        learning_rate=1e-3, weight_decay=0.1
    )
    interrupted_opt = interrupted.configure_optimizer(
        learning_rate=1e-3, weight_decay=0.1
    )
    batches = [
        (
            torch.randint(0, config.vocab_size, (1, 4)),
            torch.randint(0, config.vocab_size, (1, 4)),
        )
        for _ in range(2)
    ]

    def step(
        model: GPT,
        optimizer: torch.optim.Optimizer,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        _, loss = model(batch[0], batch[1], return_logits=False)
        assert loss is not None
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    step(reference, reference_opt, batches[0])
    step(reference, reference_opt, batches[1])
    step(interrupted, interrupted_opt, batches[0])
    checkpoint = {
        "model": interrupted.state_dict(),
        "optimizer": interrupted_opt.state_dict(),
    }
    resumed = GPT(config)
    resumed_opt = resumed.configure_optimizer(
        learning_rate=1e-3, weight_decay=0.1
    )
    resumed.load_state_dict(checkpoint["model"])
    resumed_opt.load_state_dict(checkpoint["optimizer"])
    step(resumed, resumed_opt, batches[1])
    for reference_parameter, resumed_parameter in zip(
        reference.parameters(), resumed.parameters(), strict=True
    ):
        torch.testing.assert_close(reference_parameter, resumed_parameter)
    print("model + AdamW continuation OK")


def main() -> None:
    test_loader_and_prefetched_batch()
    test_rng_round_trip()
    test_atomic_save_and_alias()
    test_strict_compatibility()
    test_model_optimizer_continuation()


if __name__ == "__main__":
    main()
