#!/usr/bin/env python3
"""CPU tests for fixed WSD, legacy LR search, WSqD, and loss-aware WSqD."""

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
        "wsqd_shift_steps": 100.0,
        "plateau_min_improvement": 0.02,
        "plateau_drop_factor": 0.80,
        "plateau_patience": 2,
        "plateau_cooldown_windows": 1,
        "plateau_min_multiplier": 0.25,
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


def scheduler_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "num_iterations": 1_000,
        "warmup_iters": 10,
        "warmdown_iters": 100,
        "learning_rate": 0.0018,
        "lr_schedule": "loss-aware-wsqd",
        "lr_search_start": 10,
        "lr_search_end": 700,
        "lr_velocity_window": 16,
        "lr_upscale_factor": 1.08,
        "lr_downscale_factor": 1.05,
        "lr_factor_decay": 0.90,
        "lr_velocity_trigger": 0.10,
        "lr_velocity_accept": 0.02,
        "lr_velocity_ema_beta": 0.80,
        "lr_loss_rise_guard": 0.10,
        "lr_min_peak": 0.0012,
        "lr_max_peak": 0.0022,
        "lr_wsqd_shift": 100.0,
        "lr_plateau_min_improvement": 0.02,
        "lr_plateau_drop_factor": 0.80,
        "lr_plateau_patience": 2,
        "lr_plateau_cooldown": 1,
        "lr_plateau_min_multiplier": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
    assert math.isclose(scheduler.lr_for_step(3_744), 0.0018)
    assert math.isclose(scheduler.lr_for_step(4_767), 0.0018 / 1_024)
    print("fixed WSD parity OK")


def test_wsqd_monotonic_and_cooldown() -> None:
    scheduler = make_scheduler(
        schedule="wsqd",
        total_steps=1_536,
        warmup_steps=256,
        warmdown_steps=320,
        search_start=256,
        search_end=1_216,
        wsqd_shift_steps=512,
    )
    assert math.isclose(scheduler.lr_for_step(255), 0.0018)
    assert math.isclose(scheduler.lr_for_step(256), 0.0018)
    values = [scheduler.lr_for_step(step) for step in range(256, 1_536)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))
    assert scheduler.phase_at(1_216) == "warmdown"
    print("WSqD monotonic base and reserved cooldown OK")


def test_legacy_adaptive_search_still_replays() -> None:
    scheduler = make_scheduler()
    step = scheduler.search_start
    step, events = feed_window(scheduler, step, 0.010)
    assert not events
    step, events = feed_window(scheduler, step, 0.004)
    assert [event.kind for event in events] == ["lr_trial_start"]
    step, events = feed_window(scheduler, step, 0.008)
    assert [event.kind for event in events] == ["lr_trial_accept"]
    print("legacy forward-only LR search remains reproducible")


def test_loss_aware_never_raises_lr() -> None:
    scheduler = make_scheduler(
        schedule="loss-aware-wsqd",
        total_steps=1_536,
        warmup_steps=256,
        warmdown_steps=320,
        search_start=256,
        search_end=1_216,
        plateau_min_improvement=0.0225,
        plateau_patience=2,
    )
    medians = [5.0720, 4.6139, 4.3683, 4.2368, 4.1488, 4.0781, 4.0192, 3.9562]
    velocities = [0.000709, 0.000480, 0.000227, 0.000240, 0.000335, 0.000138, 0.000128, 0.000066]
    step = scheduler.search_start
    lrs = [scheduler.lr_for_step(step)]
    decisions = []
    for median, velocity in zip(medians, velocities):
        start_loss = median * math.exp(velocity * (scheduler.window_steps - 1) / 2)
        step, events = feed_window(scheduler, step, velocity, start_loss=start_loss)
        decisions.extend(events)
        lrs.append(scheduler.lr_for_step(step))
    assert all(b <= a + 1e-15 for a, b in zip(lrs, lrs[1:]))
    assert decisions
    assert all(event.kind == "lr_monotonic_drop" for event in decisions)
    assert all(event.new_peak_lr < event.old_peak_lr for event in decisions)
    print("run-like plateau replay triggers only downward LR decisions")


def test_resume_replay() -> None:
    args = scheduler_args()
    original = build_lr_scheduler(args)
    events = []
    for step in range(80):
        loss = 5.0 * math.exp(-0.002 * step)
        original.observe(step, loss)
        events.append({"event": "train", "step": step, "train_loss": loss})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.jsonl"
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        restored = build_lr_scheduler(args)
        replay_scheduler_history(restored, path, next_step=80)
        assert restored.__dict__ == original.__dict__
    validate_scheduler_resume_args({"args": vars(args)}, args)
    print("adaptive state replay OK")


def test_tui_row() -> None:
    display = object.__new__(ResumableProgressDisplay)
    display._warmup_iters = 256
    display._warmdown_iters = 320
    display._total_steps = 1_536
    display._scheduler_status = {
        "name": "loss-aware-wsqd",
        "phase": "loss-monitor",
        "peak_lr": 0.0018,
        "base_lr": 0.0014,
        "current_lr": 0.00112,
        "lr_multiplier": 0.8,
        "window_progress": 17,
        "window_steps": 128,
        "velocity": 1.2e-4,
        "velocity_ema": 1.4e-4,
        "window_improvement": 0.011,
        "plateau_streak": 1,
        "plateau_patience": 2,
        "decisions": 1,
        "downward_adjustments": 1,
        "accepted_trials": 0,
        "rejected_trials": 0,
        "last_event": "monotonic LR drop",
    }
    row = display._lr_row(1_000, 0.00112).plain
    assert "loss-aware-wsqd" in row
    assert "base 0.001400" in row
    assert "×0.800" in row
    assert "streak 1/2" in row
    print("TUI loss-aware scheduler row OK")


def main() -> None:
    test_fixed_wsd_parity()
    test_wsqd_monotonic_and_cooldown()
    test_legacy_adaptive_search_still_replays()
    test_loss_aware_never_raises_lr()
    test_resume_replay()
    test_tui_row()


if __name__ == "__main__":
    main()
