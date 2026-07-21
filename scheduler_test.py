#!/usr/bin/env python3
"""CPU-only tests for fixed and loss-velocity WSD controllers."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from training.display import ResumableProgressDisplay
from training.lr_scheduler import (
    LossVelocityWSD,
    build_lr_scheduler,
    replay_scheduler_history,
    validate_scheduler_resume_args,
)


def make_scheduler(**overrides: object) -> LossVelocityWSD:
    values: dict[str, object] = {
        "schedule": "loss-velocity-wsd",
        "initial_peak_lr": 0.0018,
        "total_steps": 1_000,
        "warmup_steps": 10,
        "warmdown_steps": 100,
        "search_start": 10,
        "search_end": 700,
        "window_steps": 16,
        "upscale_factor": 1.08,
        "downscale_factor": 1.05,
        "factor_decay": 0.90,
        "trigger_ratio": 0.10,
        "accept_ratio": 0.02,
        "velocity_ema_beta": 0.80,
        "rise_ratio": 0.10,
        "min_peak_lr": 0.0012,
        "max_peak_lr": 0.0022,
    }
    values.update(overrides)
    return LossVelocityWSD(**values)  # type: ignore[arg-type]


def feed_window(
    scheduler: LossVelocityWSD,
    step: int,
    rate: float,
    *,
    start_loss: float = 5.0,
) -> tuple[int, list[object]]:
    events: list[object] = []
    for offset in range(scheduler.window_steps):
        loss = start_loss * math.exp(-rate * offset)
        events.extend(scheduler.observe(step, loss))
        step += 1
    return step, events


def test_fixed_wsd_parity() -> None:
    scheduler = make_scheduler(
        schedule="wsd",
        total_steps=4_768,
        warmup_steps=256,
        warmdown_steps=1_024,
        search_start=256,
        search_end=2_808,
    )
    assert math.isclose(scheduler.lr_for_step(0), 0.0018 / 256)
    assert math.isclose(scheduler.lr_for_step(255), 0.0018)
    assert math.isclose(scheduler.lr_for_step(256), 0.0018)
    assert math.isclose(scheduler.lr_for_step(3_744), 0.0018)
    assert math.isclose(scheduler.lr_for_step(4_767), 0.0018 / 1_024)
    print("fixed WSD parity OK")


def test_robust_velocity() -> None:
    losses = [5.0 * math.exp(-0.002 * step) for step in range(128)]
    losses[31] *= 1.25
    velocity = LossVelocityWSD.estimate_velocity(losses)
    assert velocity > 0
    print(f"robust positive loss velocity OK ({velocity:.3e}/step)")


def test_adaptive_trial_accept_and_reject() -> None:
    scheduler = make_scheduler()
    step = scheduler.search_start
    step, events = feed_window(scheduler, step, 0.010)
    assert not events
    step, events = feed_window(scheduler, step, 0.004)
    assert [event.kind for event in events] == ["lr_trial_start"]
    trial_peak = scheduler.peak_lr
    assert trial_peak > scheduler.initial_peak_lr
    step, events = feed_window(scheduler, step, 0.008)
    assert [event.kind for event in events] == ["lr_trial_accept"]
    assert scheduler.accepted_trials == 1

    scheduler.velocity_ema = 0.010
    step, events = feed_window(scheduler, step, 0.001)
    assert not events
    step, events = feed_window(scheduler, step, 0.001)
    assert [event.kind for event in events] == ["lr_trial_start"]
    origin = scheduler.trial_origin_lr
    assert origin is not None
    step, events = feed_window(scheduler, step, 0.0005)
    assert [event.kind for event in events] == ["lr_trial_reject"]
    assert scheduler.peak_lr < origin
    assert scheduler.rejected_trials == 1
    print("adaptive accept/reject search OK")


def test_log_replay_and_legacy_resume_args() -> None:
    args = SimpleNamespace(
        num_iterations=1_000,
        warmup_iters=10,
        warmdown_iters=100,
        learning_rate=0.0018,
        lr_schedule="loss-velocity-wsd",
        lr_search_start=10,
        lr_search_end=700,
        lr_velocity_window=16,
        lr_upscale_factor=1.08,
        lr_downscale_factor=1.05,
        lr_factor_decay=0.90,
        lr_velocity_trigger=0.10,
        lr_velocity_accept=0.02,
        lr_velocity_ema_beta=0.80,
        lr_loss_rise_guard=0.10,
        lr_min_peak=0.0012,
        lr_max_peak=0.0022,
    )
    original = build_lr_scheduler(args)
    events: list[dict[str, object]] = []
    for step in range(50):
        loss = 5.0 * math.exp(-0.006 * step)
        original.observe(step, loss)
        events.append({"event": "train", "step": step, "train_loss": loss})

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        restored = build_lr_scheduler(args)
        replay_scheduler_history(restored, path, next_step=50)
        assert restored.__dict__ == original.__dict__

    checkpoint = {"args": vars(args)}
    validate_scheduler_resume_args(checkpoint, args)
    legacy_wsd_args = SimpleNamespace(
        **{
            **vars(args),
            "lr_schedule": "wsd",
            "lr_search_start": None,
            "lr_search_end": None,
            "lr_velocity_window": 128,
            "lr_upscale_factor": 1.08,
            "lr_downscale_factor": 1.05,
            "lr_factor_decay": 0.90,
            "lr_velocity_trigger": 0.12,
            "lr_velocity_accept": 0.03,
            "lr_velocity_ema_beta": 0.80,
            "lr_loss_rise_guard": 0.06,
            "lr_min_peak": None,
            "lr_max_peak": None,
        }
    )
    validate_scheduler_resume_args({"args": {}}, legacy_wsd_args)
    print("loss-log replay and legacy WSD resume compatibility OK")


def test_tui_scheduler_row() -> None:
    display = object.__new__(ResumableProgressDisplay)
    display._scheduler_status = {
        "name": "loss-velocity-wsd",
        "phase": "lr-search",
        "peak_lr": 0.0018,
        "window_progress": 17,
        "window_steps": 128,
        "velocity": 1.2e-4,
        "velocity_ema": 1.4e-4,
        "decisions": 2,
        "accepted_trials": 1,
        "rejected_trials": 1,
        "last_event": "accepted higher LR",
    }
    display._warmup_iters = 256
    display._warmdown_iters = 512
    display._total_steps = 4_768
    row = display._lr_row(1_000, 0.0018).plain
    assert "loss-velocity-wsd" in row
    assert "window 17/128" in row
    assert "accepted higher LR" in row
    print("TUI scheduler status row OK")


def main() -> None:
    test_fixed_wsd_parity()
    test_robust_velocity()
    test_adaptive_trial_accept_and_reject()
    test_log_replay_and_legacy_resume_args()
    test_tui_scheduler_row()


if __name__ == "__main__":
    main()
