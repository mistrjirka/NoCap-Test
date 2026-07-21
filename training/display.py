"""Resume- and scheduler-aware wrapper around the Rich TTY dashboard."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.text import Text

from progress import ProgressDisplay


class ResumableProgressDisplay(ProgressDisplay):
    def __init__(
        self,
        *,
        append_log: bool,
        elapsed_offset_s: float,
        initial_train_losses: list[float],
        initial_val_losses: list[tuple[int, float]],
        log_path: Path | None,
        **kwargs: object,
    ) -> None:
        super().__init__(log_path=None, **kwargs)  # type: ignore[arg-type]
        now = time.perf_counter()
        self._start_time = now - max(0.0, elapsed_offset_s)
        self._last_step_ts = now
        self._train_losses = list(initial_train_losses)
        self._val_losses = list(initial_val_losses)
        self._best_val = (
            min(self._val_losses, key=lambda item: item[1])
            if self._val_losses
            else None
        )
        self._target_reached = bool(
            self._best_val
            and self._best_val[0] > 0
            and self._best_val[1] <= self._target_val_loss
        )
        self._scheduler_status: dict[str, object] | None = None
        if log_path is not None:
            mode = "a" if append_log else "w"
            self._log_console = Console(
                file=log_path.open(mode, encoding="utf-8"), width=999
            )

    def reset_step_timer(self) -> None:
        """Exclude validation, checkpoint I/O, and downtime from TTY step timing."""
        self._last_step_ts = time.perf_counter()

    def set_scheduler_status(self, status: dict[str, object]) -> None:
        self._scheduler_status = dict(status)

    def _lr_row(self, step: int, lr: float) -> Text:
        status = self._scheduler_status
        if status is None:
            return super()._lr_row(step, lr)

        phase = str(status.get("phase", "stable"))
        if phase == "warmup":
            pct = (step + 1) / max(self._warmup_iters, 1)
            phase_style = "yellow"
        elif phase == "warmdown":
            steps_in = step - (self._total_steps - self._warmdown_iters)
            pct = 1.0 - steps_in / max(self._warmdown_iters, 1)
            phase_style = "red"
        elif phase == "lr-trial":
            pct = 1.0
            phase_style = "bold yellow"
        elif phase == "lr-search":
            pct = 1.0
            phase_style = "bold cyan"
        else:
            pct = 1.0
            phase_style = "green"
        pct = max(0.0, min(1.0, pct))

        bar_w = 30
        filled = int(pct * bar_w)
        text = Text()
        text.append("learning rate  ", style="bright_black")
        text.append("█" * filled, style="cyan")
        text.append("░" * (bar_w - filled), style="bright_black cyan")
        text.append(f"  {lr:.6f}  [", style="bright_black")
        text.append(phase, style=phase_style)
        text.append("]")

        name = str(status.get("name", "wsd"))
        peak = float(status.get("peak_lr", lr))
        text.append("\nLR controller  ", style="bright_black")
        text.append(name, style="bold cyan" if name != "wsd" else "white")
        text.append(f"  peak {peak:.6f}", style="white")

        if name == "loss-velocity-wsd":
            progress = int(status.get("window_progress", 0))
            window = int(status.get("window_steps", 0))
            decisions = int(status.get("decisions", 0))
            accepted = int(status.get("accepted_trials", 0))
            rejected = int(status.get("rejected_trials", 0))
            velocity = status.get("velocity")
            velocity_ema = status.get("velocity_ema")
            text.append(f"  window {progress}/{window}", style="bright_black")
            if velocity is not None:
                text.append(f"  v={float(velocity):.2e}", style="white")
            if velocity_ema is not None:
                text.append(f"  ema={float(velocity_ema):.2e}", style="bright_black")
            text.append(
                f"  decisions {decisions} ({accepted}✓/{rejected}×)",
                style="bright_black",
            )
            last_event = str(status.get("last_event", ""))
            if last_event:
                text.append(f"\nlast LR event  {last_event}", style="bright_black")
        return text
