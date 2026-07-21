"""Checkpoint helpers for exact single-GPU NoCap training resumes."""

from __future__ import annotations

import json
import os
import random
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loader import DistributedDataLoader, _load_data_shard

CHECKPOINT_FORMAT_VERSION = 2

CRITICAL_ARGUMENTS = (
    "input_bin",
    "input_val_bin",
    "preset",
    "embedding_dim",
    "activation",
    "embedding_projection",
    "qk_norm",
    "conv_layers",
    "conv_kernel_size",
    "batch_size",
    "grad_accumulation_steps",
    "sequence_length",
    "context_schedule",
    "num_iterations",
    "learning_rate",
    "warmup_iters",
    "warmdown_iters",
    "weight_decay",
    "val_loss_every",
    "val_batch_size",
    "validation_tokens",
    "target_val_loss",
    "stop_at_target",
    "compile",
    "seed",
)


def resolve_resume_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        for filename in ("latest.pt", "final.pt"):
            candidate = path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Resume directory {path} contains neither latest.pt nor final.pt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def loader_state_dict(loader: DistributedDataLoader) -> dict[str, object]:
    return {
        "files": [str(Path(path).resolve()) for path in loader.files],
        "current_shard": loader.current_shard,
        "current_position": loader.current_position,
        "batch_size": loader.batch_size,
        "sequence_length": loader.sequence_length,
        "process_rank": loader.process_rank,
        "num_processes": loader.num_processes,
    }


def load_loader_state(
    loader: DistributedDataLoader, state: dict[str, object]
) -> None:
    expected_files = [str(Path(path).resolve()) for path in loader.files]
    saved_files = [str(path) for path in state["files"]]  # type: ignore[index]
    if saved_files != expected_files:
        raise ValueError("Training shard list/order differs from the checkpoint")
    if int(state["process_rank"]) != loader.process_rank:
        raise ValueError("Data-loader process rank differs from the checkpoint")
    if int(state["num_processes"]) != loader.num_processes:
        raise ValueError("Data-loader world size differs from the checkpoint")

    batch_size = int(state["batch_size"])
    sequence_length = int(state["sequence_length"])
    current_shard = int(state["current_shard"])
    current_position = int(state["current_position"])
    if not 0 <= current_shard < len(loader.files):
        raise ValueError(f"Invalid checkpoint shard index: {current_shard}")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Invalid checkpoint batch shape")

    loader.batch_size = batch_size
    loader.sequence_length = sequence_length
    loader.current_shard = current_shard
    loader.tokens = _load_data_shard(loader.files[current_shard])
    if not 0 <= current_position <= len(loader.tokens):
        raise ValueError(f"Invalid checkpoint token position: {current_position}")
    loader.current_position = current_position


def _normalise(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    return value


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    args: Namespace,
    *,
    model_config: dict[str, Any],
    amp_name: str,
    world_size: int,
    gpu_name: str,
    torch_version: str,
) -> None:
    version = int(checkpoint.get("format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format {version} cannot be resumed exactly; expected "
            f"format {CHECKPOINT_FORMAT_VERSION}. Older checkpoints lack loader, "
            "prefetch, GradScaler, and RNG state."
        )

    if _normalise(checkpoint.get("model_config")) != _normalise(model_config):
        raise ValueError("Resume model configuration differs from the checkpoint")

    saved_args = checkpoint.get("args", {})
    mismatches: list[str] = []
    for name in CRITICAL_ARGUMENTS:
        saved = _normalise(saved_args.get(name))
        current = _normalise(getattr(args, name))
        if saved != current:
            mismatches.append(f"{name}: checkpoint={saved!r}, current={current!r}")

    runtime = checkpoint.get("runtime", {})
    for name, saved, current in (
        ("amp", runtime.get("amp"), amp_name),
        ("world_size", runtime.get("world_size"), world_size),
        ("gpu", runtime.get("gpu"), gpu_name),
        ("torch_version", runtime.get("torch_version"), torch_version),
    ):
        if saved != current:
            mismatches.append(
                f"runtime.{name}: checkpoint={saved!r}, current={current!r}"
            )

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Exact resume rejected because benchmark-critical settings changed:\n"
            f"  - {details}"
        )

    next_step = int(checkpoint.get("next_step", -1))
    if not 0 <= next_step <= args.num_iterations:
        raise ValueError(
            f"Checkpoint next_step={next_step} is outside [0, {args.num_iterations}]"
        )

    required = (
        "model",
        "optimizer",
        "scaler",
        "train_loader",
        "prefetched_batch",
        "rng_state",
        "training_time_ms",
        "run_id",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"Resume checkpoint is missing required fields: {missing}")


def load_display_history(
    path: Path,
) -> tuple[list[float], list[tuple[int, float]]]:
    train_losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    if not path.exists():
        return train_losses, val_losses
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "train" and "train_loss" in event:
                train_losses.append(float(event["train_loss"]))
            elif event.get("event") == "validation" and "val_loss" in event:
                val_losses.append((int(event["step"]), float(event["val_loss"])))
    return train_losses, val_losses
