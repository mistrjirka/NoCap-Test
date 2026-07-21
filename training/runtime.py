"""Runtime construction, timing, signals, and checkpoint publication."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from data_loader import DistributedDataLoader
from model import GPT, GPTConfig
from resume_utils import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_link_or_copy,
    atomic_torch_save,
    capture_rng_state,
    loader_state_dict,
)
from training.display import ResumableProgressDisplay


def print0(*args: object, **kwargs: object) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def json_log(path: Path | None, event: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class Runtime:
    args: argparse.Namespace
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    local_grad_accum: int
    device: torch.device
    amp_name: str
    amp_dtype: torch.dtype
    scaler: Any
    config: GPTConfig
    train_loader: DistributedDataLoader
    val_loader: DistributedDataLoader
    base_model: GPT
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    parameter_count: int
    mixer_summary: str
    microbatch_tokens: int
    val_microbatch_tokens: int
    tokens_per_iteration: int
    start_step: int
    current_t: int
    training_time_ms: float
    run_id: str
    wandb_id: str
    validation_completed_for_step: int | None
    last_completed_validation_step: int | None
    x: torch.Tensor
    y: torch.Tensor
    run_dir: Path
    log_path: Path | None
    display: ResumableProgressDisplay
    checkpoint: dict[str, Any] | None
    wandb: Any | None = None

    timer_active: bool = False
    interval_start: float = 0.0
    stop_requested: bool = False
    signal_count: int = 0
    previous_handlers: dict[int, Any] = field(default_factory=dict)
    safe_boundary: bool = True
    safe_next_step: int = 0
    target_reached: bool = False
    interrupted: bool = False
    final_step: int = 0
    final_checkpoint_written: bool = False

    def amp_context(self):
        return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)

    def resume_timer(self) -> None:
        if not self.timer_active:
            self.interval_start = time.perf_counter()
            self.timer_active = True

    def pause_timer(self) -> None:
        if self.timer_active:
            torch.cuda.synchronize()
            self.training_time_ms += 1000.0 * (
                time.perf_counter() - self.interval_start
            )
            self.timer_active = False

    def measured_time_ms(self) -> float:
        if not self.timer_active:
            return self.training_time_ms
        torch.cuda.synchronize()
        return self.training_time_ms + 1000.0 * (
            time.perf_counter() - self.interval_start
        )

    def checkpoint_payload(
        self,
        next_step: int,
        *,
        reason: str,
        validation_done_step: int | None,
    ) -> dict[str, Any]:
        if not self.safe_boundary:
            raise RuntimeError("Refusing to checkpoint a partially accumulated step")
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": self.base_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "next_step": next_step,
            "args": vars(self.args),
            "model_config": self.config.to_dict(),
            "runtime": {
                "amp": self.amp_name,
                "world_size": self.world_size,
                "gpu": torch.cuda.get_device_name(self.device),
                "torch_version": torch.__version__,
            },
            "training_time_ms": self.training_time_ms,
            "train_loader": loader_state_dict(self.train_loader),
            "prefetched_batch": {
                "x": self.x.detach().cpu(),
                "y": self.y.detach().cpu(),
            },
            "rng_state": capture_rng_state(),
            "run_id": self.run_id,
            "wandb_id": self.wandb_id,
            "target_reached": self.target_reached,
            "validation_completed_for_step": validation_done_step,
            "reason": reason,
            "saved_at": time.time(),
        }

    def save_checkpoint(
        self,
        next_step: int,
        *,
        reason: str,
        validation_done_step: int | None = None,
        final: bool = False,
        resume_after: bool = True,
    ) -> None:
        was_active = self.timer_active
        self.pause_timer()
        if self.rank == 0:
            latest = self.run_dir / "latest.pt"
            atomic_torch_save(
                self.checkpoint_payload(
                    next_step,
                    reason=reason,
                    validation_done_step=validation_done_step,
                ),
                latest,
            )
            if self.args.keep_step_checkpoints and reason == "periodic":
                atomic_link_or_copy(
                    latest, self.run_dir / f"step-{next_step:06d}.pt"
                )
            if final:
                atomic_link_or_copy(latest, self.run_dir / "final.pt")
                self.final_checkpoint_written = True
            json_log(
                self.log_path,
                {
                    "event": "checkpoint",
                    "next_step": next_step,
                    "reason": reason,
                    "training_time_s": self.training_time_ms / 1000.0,
                    "path": "final.pt" if final else "latest.pt",
                },
            )
            print0(
                f"Checkpoint saved: "
                f"{self.run_dir / ('final.pt' if final else 'latest.pt')} "
                f"(next step {next_step})"
            )
        if self.distributed:
            dist.barrier()
        self.display.reset_step_timer()
        if resume_after and was_active:
            self.resume_timer()

    def request_stop(self, signum: int, _frame: object) -> None:
        self.signal_count += 1
        if self.signal_count == 1:
            self.stop_requested = True
            print0(
                f"Received signal {signum}; finishing the current optimizer step "
                "and writing an exact checkpoint. Send it again to abort immediately."
            )
        else:
            raise KeyboardInterrupt

    def install_signal_handlers(self) -> None:
        self.previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in self.previous_handlers:
            signal.signal(signum, self.request_stop)

    def restore_signal_handlers(self) -> None:
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)
