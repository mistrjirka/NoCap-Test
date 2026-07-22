"""Learning-rate schedules for NoCap pretraining.

Supported controllers:

* ``wsd``: the original fixed Warmup–Stable–Decay schedule.
* ``loss-velocity-wsd``: the earlier AdaLRS-inspired peak-LR search. It is kept
  for reproducibility, but its forward-only up-trials are experimental.
* ``wsqd``: a paper-inspired shifted inverse-square-root base with a final
  linear cooldown.
* ``loss-aware-wsqd``: WSqD plus conservative, downward-only loss-triggered
  multiplier reductions.

The loss-aware controller never raises the learning rate after warmup. This is
intentional: AdaLRS relies on trial-state backtracking, and its own ablation
reports that removing backtracking can leave persistent damage after an
oversized LR trial. Model/optimizer backtracking is undesirable in a timed
speedrun, so the safer adaptation is monotonic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ScheduleName = Literal[
    "wsd",
    "loss-velocity-wsd",
    "wsqd",
    "loss-aware-wsqd",
]


@dataclass(frozen=True)
class SchedulerEvent:
    kind: str
    step: int
    old_peak_lr: float
    new_peak_lr: float
    reference_velocity: float | None = None
    observed_velocity: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LossVelocityWSD:
    """Backwards-compatible controller implementing all supported schedules."""

    def __init__(
        self,
        *,
        schedule: ScheduleName,
        initial_peak_lr: float,
        total_steps: int,
        warmup_steps: int,
        warmdown_steps: int,
        search_start: int,
        search_end: int,
        window_steps: int,
        upscale_factor: float,
        downscale_factor: float,
        factor_decay: float,
        trigger_ratio: float,
        accept_ratio: float,
        velocity_ema_beta: float,
        rise_ratio: float,
        min_peak_lr: float,
        max_peak_lr: float,
        wsqd_shift_steps: float = 512.0,
        plateau_min_improvement: float = 0.02,
        plateau_drop_factor: float = 0.80,
        plateau_patience: int = 2,
        plateau_cooldown_windows: int = 1,
        plateau_min_multiplier: float = 0.25,
    ) -> None:
        self.schedule = schedule
        self.initial_peak_lr = initial_peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmdown_steps = warmdown_steps
        self.search_start = search_start
        self.search_end = search_end
        self.window_steps = window_steps
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor
        self.factor_decay = factor_decay
        self.trigger_ratio = trigger_ratio
        self.accept_ratio = accept_ratio
        self.velocity_ema_beta = velocity_ema_beta
        self.rise_ratio = rise_ratio
        self.min_peak_lr = min_peak_lr
        self.max_peak_lr = max_peak_lr

        self.wsqd_shift_steps = float(wsqd_shift_steps)
        self.plateau_min_improvement = plateau_min_improvement
        self.plateau_drop_factor = plateau_drop_factor
        self.plateau_patience = plateau_patience
        self.plateau_cooldown_windows = plateau_cooldown_windows
        self.plateau_min_multiplier = plateau_min_multiplier
        self._validate()

        # Existing loss-velocity WSD state.
        self.peak_lr = initial_peak_lr
        self.mode = "monitor"
        self.window_losses: list[float] = []
        self.velocity_ema: float | None = None
        self.last_velocity: float | None = None
        self.last_window_loss: float | None = None
        self.reference_velocity: float | None = None
        self.reference_loss: float | None = None
        self.trial_origin_lr: float | None = None
        self.current_upscale = upscale_factor
        self.current_downscale = downscale_factor
        self.cooldown_windows = 0
        self.consecutive_rising_windows = 0
        self.decisions = 0
        self.accepted_trials = 0
        self.rejected_trials = 0
        self.last_event = ""

        # Downward-only loss-aware WSqD state.
        self.lr_multiplier = 1.0
        self.plateau_streak = 0
        self.plateau_cooldown_remaining = 0
        self.last_window_improvement: float | None = None
        self.downward_adjustments = 0

    def _validate(self) -> None:
        stable_end = self.total_steps - self.warmdown_steps
        if self.schedule not in {
            "wsd",
            "loss-velocity-wsd",
            "wsqd",
            "loss-aware-wsqd",
        }:
            raise ValueError(f"Unsupported LR schedule: {self.schedule}")
        if self.initial_peak_lr <= 0 or self.total_steps <= 0:
            raise ValueError("Peak LR and total steps must be positive")
        if not 0 <= self.warmup_steps <= stable_end:
            raise ValueError("Warmup overlaps terminal warmdown")
        if not self.warmup_steps <= self.search_start <= self.search_end <= stable_end:
            raise ValueError("LR observation interval must lie between warmup and warmdown")
        if self.window_steps < 8:
            raise ValueError("Loss window must contain at least 8 steps")
        if self.upscale_factor <= 1 or self.downscale_factor <= 1:
            raise ValueError("LR scale factors must exceed 1")
        if not 0 < self.factor_decay <= 1:
            raise ValueError("Factor decay must be in (0, 1]")
        if not 0 <= self.trigger_ratio < 1 or not 0 <= self.accept_ratio < 1:
            raise ValueError("Velocity ratios must be in [0, 1)")
        if not 0 <= self.velocity_ema_beta < 1 or self.rise_ratio < 0:
            raise ValueError("Invalid EMA or loss-rise guard")
        if not 0 < self.min_peak_lr <= self.initial_peak_lr <= self.max_peak_lr:
            raise ValueError("Peak-LR bounds must contain the initial peak")
        if self.wsqd_shift_steps < 0:
            raise ValueError("WSqD shift must be non-negative")
        if self.plateau_min_improvement < 0:
            raise ValueError("Plateau minimum improvement cannot be negative")
        if not 0 < self.plateau_drop_factor < 1:
            raise ValueError("Plateau drop factor must be in (0, 1)")
        if self.plateau_patience < 1:
            raise ValueError("Plateau patience must be at least 1 window")
        if self.plateau_cooldown_windows < 0:
            raise ValueError("Plateau cooldown cannot be negative")
        if not 0 < self.plateau_min_multiplier <= 1:
            raise ValueError("Minimum LR multiplier must be in (0, 1]")

    @property
    def stable_end(self) -> int:
        return self.total_steps - self.warmdown_steps

    @property
    def uses_wsqd_base(self) -> bool:
        return self.schedule in {"wsqd", "loss-aware-wsqd"}

    def phase_at(self, step: int) -> str:
        if self.warmup_steps and step < self.warmup_steps:
            return "warmup"
        if self.warmdown_steps and step >= self.stable_end:
            return "warmdown"
        if self.schedule == "loss-velocity-wsd" and self.search_start <= step < self.search_end:
            return "lr-trial" if self.mode == "trial" else "lr-search"
        if self.schedule == "loss-aware-wsqd" and self.search_start <= step < self.search_end:
            if self.plateau_cooldown_remaining:
                return "loss-cooldown"
            return "loss-monitor"
        if self.uses_wsqd_base:
            return "sqrt-decay"
        return "stable"

    def _wsqd_base_lr(self, step: int) -> float:
        """Shifted inverse-square-root LR, normalized to peak after warmup."""
        post_warmup_step = max(1, step - self.warmup_steps + 1)
        numerator = self.wsqd_shift_steps + 1.0
        denominator = self.wsqd_shift_steps + float(post_warmup_step)
        return self.initial_peak_lr * math.sqrt(numerator / denominator)

    def base_lr_for_step(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.initial_peak_lr * (step + 1) / self.warmup_steps
        if not self.uses_wsqd_base:
            return self.peak_lr

        if self.warmdown_steps and step >= self.stable_end:
            anchor = self._wsqd_base_lr(self.stable_end)
            return anchor * max(0, self.total_steps - step) / self.warmdown_steps
        return self._wsqd_base_lr(step)

    def lr_for_step(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.initial_peak_lr * (step + 1) / self.warmup_steps

        if self.uses_wsqd_base:
            multiplier = self.lr_multiplier if self.schedule == "loss-aware-wsqd" else 1.0
            return self.base_lr_for_step(step) * multiplier

        if self.warmdown_steps and step >= self.stable_end:
            return self.peak_lr * max(0, self.total_steps - step) / self.warmdown_steps
        return self.peak_lr

    @staticmethod
    def estimate_velocity(losses: list[float]) -> float:
        """Robust relative loss descent per step from a log-loss least-squares fit."""
        clean = [float(x) for x in losses if math.isfinite(x) and x > 0]
        if len(clean) < 4:
            return float("nan")
        ordered = sorted(clean)
        lo = ordered[int(0.05 * (len(ordered) - 1))]
        hi = ordered[int(0.95 * (len(ordered) - 1))]
        clipped = [min(max(x, lo), hi) for x in clean]
        smoothed: list[float] = []
        value = clipped[0]
        for item in clipped:
            value = 0.9 * value + 0.1 * item
            smoothed.append(math.log(max(value, 1e-12)))
        n = len(smoothed)
        xm = (n - 1) / 2
        ym = sum(smoothed) / n
        num = sum((i - xm) * (y - ym) for i, y in enumerate(smoothed))
        den = sum((i - xm) ** 2 for i in range(n))
        return -num / max(den, 1e-12)

    @staticmethod
    def representative_loss(losses: list[float]) -> float:
        clean = sorted(float(x) for x in losses if math.isfinite(x))
        if not clean:
            return float("nan")
        middle = len(clean) // 2
        return clean[middle] if len(clean) % 2 else (clean[middle - 1] + clean[middle]) / 2

    def _set_peak(self, value: float) -> float:
        self.peak_lr = min(max(value, self.min_peak_lr), self.max_peak_lr)
        return self.peak_lr

    def _shrink_factors(self) -> None:
        self.current_upscale = 1 + (self.current_upscale - 1) * self.factor_decay
        self.current_downscale = 1 + (self.current_downscale - 1) * self.factor_decay

    def reset_observations(self, *, cancel_trial: bool = True) -> None:
        self.window_losses.clear()
        self.velocity_ema = self.last_velocity = self.last_window_loss = None
        self.consecutive_rising_windows = 0
        self.cooldown_windows = 1
        self.plateau_streak = 0
        self.plateau_cooldown_remaining = 1
        self.last_window_improvement = None
        if cancel_trial and self.mode == "trial":
            self._set_peak(self.trial_origin_lr or self.peak_lr)
            self.mode = "monitor"
            self.reference_velocity = self.reference_loss = self.trial_origin_lr = None
            self.last_event = "cancelled LR trial after context change"

    def _monitor(self, step: int, velocity: float, median_loss: float) -> list[SchedulerEvent]:
        previous_ema, previous_loss = self.velocity_ema, self.last_window_loss
        rising = previous_loss is not None and median_loss > previous_loss * (1 + self.rise_ratio)
        self.consecutive_rising_windows = self.consecutive_rising_windows + 1 if rising else 0
        self.velocity_ema = velocity if previous_ema is None else (
            self.velocity_ema_beta * previous_ema + (1 - self.velocity_ema_beta) * velocity
        )
        events: list[SchedulerEvent] = []
        if self.cooldown_windows:
            self.cooldown_windows -= 1
        elif self.consecutive_rising_windows >= 2:
            old = self.peak_lr
            new = self._set_peak(old / self.current_downscale)
            if new < old:
                self.decisions += 1
                self.last_event = "downscaled after two rising loss windows"
                events.append(SchedulerEvent(
                    "lr_downscale", step, old, new, previous_ema, velocity,
                    "two consecutive rising median-loss windows",
                ))
                self._shrink_factors()
            self.consecutive_rising_windows = 0
            self.cooldown_windows = 1
            self.velocity_ema = velocity
        elif previous_ema and previous_ema > 0 and velocity < previous_ema * (1 - self.trigger_ratio):
            candidate = min(self.peak_lr * self.current_upscale, self.max_peak_lr)
            if candidate > self.peak_lr * (1 + 1e-12):
                old = self.peak_lr
                self.trial_origin_lr = old
                self.reference_velocity = velocity
                self.reference_loss = median_loss
                self.mode = "trial"
                self._set_peak(candidate)
                self.last_event = "started higher-LR velocity trial"
                events.append(SchedulerEvent(
                    "lr_trial_start", step, old, self.peak_lr, velocity, None,
                    "loss descent velocity fell below its smoothed reference",
                ))
        self.last_velocity, self.last_window_loss = velocity, median_loss
        return events

    def _finish_trial(self, step: int, velocity: float, median_loss: float) -> list[SchedulerEvent]:
        origin, reference = self.trial_origin_lr, self.reference_velocity
        if origin is None or reference is None:
            raise RuntimeError("Incomplete adaptive LR trial state")
        loss_ok = self.reference_loss is None or median_loss <= self.reference_loss * (1 + self.rise_ratio)
        improved = velocity > reference * (1 + self.accept_ratio)
        old = self.peak_lr
        if improved and loss_ok:
            self.accepted_trials += 1
            event = SchedulerEvent(
                "lr_trial_accept", step, origin, self.peak_lr, reference, velocity,
                "higher LR improved robust loss descent velocity",
            )
            self.last_event = "accepted higher LR"
        else:
            self.rejected_trials += 1
            new = self._set_peak(origin / self.current_downscale)
            event = SchedulerEvent(
                "lr_trial_reject", step, old, new, reference, velocity,
                "trial raised representative loss" if not loss_ok
                else "velocity gain did not exceed acceptance margin",
            )
            self.last_event = "rejected higher LR and downscaled"
        self.decisions += 1
        self._shrink_factors()
        self.mode = "monitor"
        self.cooldown_windows = 1
        self.consecutive_rising_windows = 0
        self.velocity_ema = max(velocity, reference)
        self.last_velocity, self.last_window_loss = velocity, median_loss
        self.reference_velocity = self.reference_loss = self.trial_origin_lr = None
        return [event]

    def _observe_loss_aware_wsqd(
        self,
        step: int,
        velocity: float,
        median_loss: float,
    ) -> list[SchedulerEvent]:
        previous_loss = self.last_window_loss
        previous_ema = self.velocity_ema
        improvement: float | None = None
        if previous_loss is not None and previous_loss > 0:
            improvement = (previous_loss - median_loss) / previous_loss

        self.last_window_improvement = improvement
        self.last_velocity = velocity
        self.velocity_ema = velocity if previous_ema is None else (
            self.velocity_ema_beta * previous_ema
            + (1 - self.velocity_ema_beta) * velocity
        )
        self.last_window_loss = median_loss

        if previous_loss is None:
            return []

        if self.plateau_cooldown_remaining:
            self.plateau_cooldown_remaining -= 1
            self.plateau_streak = 0
            return []

        velocity_slow = (
            previous_ema is None
            or previous_ema <= 0
            or velocity <= 0
            or velocity < previous_ema * (1 - self.trigger_ratio)
        )
        low_progress = improvement < self.plateau_min_improvement
        rising = improvement < -self.rise_ratio
        stalled = rising or (low_progress and velocity_slow)

        self.plateau_streak = self.plateau_streak + 1 if stalled else 0
        if self.plateau_streak < self.plateau_patience:
            return []

        old_multiplier = self.lr_multiplier
        new_multiplier = max(
            self.plateau_min_multiplier,
            old_multiplier * self.plateau_drop_factor,
        )
        self.plateau_streak = 0
        self.plateau_cooldown_remaining = self.plateau_cooldown_windows
        self.velocity_ema = velocity

        if new_multiplier >= old_multiplier * (1 - 1e-12):
            self.last_event = "plateau detected, but minimum LR multiplier was reached"
            return []

        self.lr_multiplier = new_multiplier
        self.decisions += 1
        self.downward_adjustments += 1
        self.last_event = (
            f"monotonic LR drop ×{self.plateau_drop_factor:.3f} after "
            f"{self.plateau_patience} low-progress windows"
        )
        decision_base_lr = self.base_lr_for_step(step + 1)
        old_lr = decision_base_lr * old_multiplier
        new_lr = decision_base_lr * new_multiplier
        reason = (
            f"median-loss improvement {improvement:.3%} below "
            f"{self.plateau_min_improvement:.3%}"
        )
        if rising:
            reason = f"representative loss rose by {-improvement:.3%}"
        return [
            SchedulerEvent(
                "lr_monotonic_drop",
                step,
                old_lr,
                new_lr,
                previous_ema,
                velocity,
                reason,
            )
        ]

    def observe(self, step: int, loss: float) -> list[SchedulerEvent]:
        adaptive = self.schedule in {"loss-velocity-wsd", "loss-aware-wsqd"}
        if not adaptive or not self.search_start <= step < self.search_end:
            return []
        if not math.isfinite(loss) or loss <= 0:
            return []

        self.window_losses.append(float(loss))
        if len(self.window_losses) < self.window_steps:
            if (
                self.schedule == "loss-velocity-wsd"
                and step + 1 >= self.search_end
                and self.mode == "trial"
            ):
                old = self.peak_lr
                self._set_peak(self.trial_origin_lr or old)
                self.mode = "monitor"
                self.reference_velocity = self.reference_loss = self.trial_origin_lr = None
                self.window_losses.clear()
                self.last_event = "cancelled incomplete LR trial at search end"
                return [SchedulerEvent(
                    "lr_trial_cancel", step, old, self.peak_lr, reason=
                    "adaptive search ended before the trial window completed",
                )]
            return []

        losses, self.window_losses = self.window_losses, []
        velocity = self.estimate_velocity(losses)
        median_loss = self.representative_loss(losses)
        if not math.isfinite(velocity):
            return []

        if self.schedule == "loss-aware-wsqd":
            return self._observe_loss_aware_wsqd(step, velocity, median_loss)
        return (
            self._finish_trial(step, velocity, median_loss)
            if self.mode == "trial"
            else self._monitor(step, velocity, median_loss)
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "schedule": self.schedule,
            "initial_peak_lr": self.initial_peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmdown_steps": self.warmdown_steps,
            "search_start": self.search_start,
            "search_end": self.search_end,
            "window_steps": self.window_steps,
            "upscale_factor": self.upscale_factor,
            "downscale_factor": self.downscale_factor,
            "factor_decay": self.factor_decay,
            "trigger_ratio": self.trigger_ratio,
            "accept_ratio": self.accept_ratio,
            "velocity_ema_beta": self.velocity_ema_beta,
            "rise_ratio": self.rise_ratio,
            "min_peak_lr": self.min_peak_lr,
            "max_peak_lr": self.max_peak_lr,
            "wsqd_shift_steps": self.wsqd_shift_steps,
            "plateau_min_improvement": self.plateau_min_improvement,
            "plateau_drop_factor": self.plateau_drop_factor,
            "plateau_patience": self.plateau_patience,
            "plateau_cooldown_windows": self.plateau_cooldown_windows,
            "plateau_min_multiplier": self.plateau_min_multiplier,
        }

    def status(self, step: int) -> dict[str, object]:
        return {
            "name": self.schedule,
            "phase": self.phase_at(step),
            "peak_lr": self.peak_lr,
            "base_lr": self.base_lr_for_step(step),
            "current_lr": self.lr_for_step(step),
            "lr_multiplier": self.lr_multiplier,
            "window_progress": len(self.window_losses),
            "window_steps": self.window_steps,
            "velocity": self.last_velocity,
            "velocity_ema": self.velocity_ema,
            "window_improvement": self.last_window_improvement,
            "plateau_streak": self.plateau_streak,
            "plateau_patience": self.plateau_patience,
            "decisions": self.decisions,
            "downward_adjustments": self.downward_adjustments,
            "accepted_trials": self.accepted_trials,
            "rejected_trials": self.rejected_trials,
            "last_event": self.last_event,
        }


def build_lr_scheduler(args: Any) -> LossVelocityWSD:
    stable_end = args.num_iterations - args.warmdown_iters
    start = args.lr_search_start if args.lr_search_start is not None else args.warmup_iters
    end = args.lr_search_end if args.lr_search_end is not None else max(
        start, int(stable_end * 0.75)
    )
    minimum = args.lr_min_peak if args.lr_min_peak is not None else args.learning_rate * 0.67
    maximum = args.lr_max_peak if args.lr_max_peak is not None else args.learning_rate * 1.25
    return LossVelocityWSD(
        schedule=args.lr_schedule,
        initial_peak_lr=args.learning_rate,
        total_steps=args.num_iterations,
        warmup_steps=args.warmup_iters,
        warmdown_steps=args.warmdown_iters,
        search_start=start,
        search_end=end,
        window_steps=args.lr_velocity_window,
        upscale_factor=args.lr_upscale_factor,
        downscale_factor=args.lr_downscale_factor,
        factor_decay=args.lr_factor_decay,
        trigger_ratio=args.lr_velocity_trigger,
        accept_ratio=args.lr_velocity_accept,
        velocity_ema_beta=args.lr_velocity_ema_beta,
        rise_ratio=args.lr_loss_rise_guard,
        min_peak_lr=minimum,
        max_peak_lr=maximum,
        wsqd_shift_steps=args.lr_wsqd_shift,
        plateau_min_improvement=args.lr_plateau_min_improvement,
        plateau_drop_factor=args.lr_plateau_drop_factor,
        plateau_patience=args.lr_plateau_patience,
        plateau_cooldown_windows=args.lr_plateau_cooldown,
        plateau_min_multiplier=args.lr_plateau_min_multiplier,
    )


SCHEDULER_DEFAULTS: dict[str, object] = {
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
    "lr_wsqd_shift": 512.0,
    "lr_plateau_min_improvement": 0.02,
    "lr_plateau_drop_factor": 0.80,
    "lr_plateau_patience": 2,
    "lr_plateau_cooldown": 1,
    "lr_plateau_min_multiplier": 0.25,
}


def validate_scheduler_resume_args(checkpoint: dict[str, Any], args: Any) -> None:
    saved_args = checkpoint.get("args", {})
    mismatches = []
    for name, default in SCHEDULER_DEFAULTS.items():
        saved, current = saved_args.get(name, default), getattr(args, name, default)
        if saved != current:
            mismatches.append(f"{name}: checkpoint={saved!r}, current={current!r}")
    if mismatches:
        raise ValueError(
            "Exact resume rejected because LR-controller settings changed:\n  - "
            + "\n  - ".join(mismatches)
        )


def replay_scheduler_history(
    scheduler: LossVelocityWSD,
    log_path: Any,
    *,
    next_step: int,
) -> None:
    if scheduler.schedule in {"wsd", "wsqd"}:
        return
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError("Adaptive exact resume requires the original events.jsonl")
    losses: dict[int, float] = {}
    context_switches: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            step, kind = int(event.get("step", -1)), event.get("event")
            if kind == "train" and 0 <= step < next_step:
                losses[step] = float(event["train_loss"])
            elif kind == "context_switch" and 0 <= step <= next_step:
                context_switches.add(step)
    for step in range(next_step):
        if step in context_switches:
            scheduler.reset_observations(cancel_trial=True)
        if step not in losses:
            raise ValueError(f"Adaptive resume is missing train loss at step {step}")
        scheduler.observe(step, losses[step])


def attach_lr_scheduler(runtime: Any) -> LossVelocityWSD:
    from training.runtime import json_log

    scheduler = build_lr_scheduler(runtime.args)
    if runtime.checkpoint is not None:
        validate_scheduler_resume_args(runtime.checkpoint, runtime.args)
        replay_scheduler_history(scheduler, runtime.log_path, next_step=runtime.start_step)
    runtime.lr_scheduler = scheduler
    runtime.display.set_scheduler_status(scheduler.status(runtime.start_step))
    json_log(
        runtime.log_path,
        {
            "event": (
                "lr_scheduler_restored"
                if runtime.checkpoint
                else "lr_scheduler_config"
            ),
            "next_step": runtime.start_step,
            "config": scheduler.config_dict(),
            "status": scheduler.status(runtime.start_step),
        },
    )
    return scheduler
