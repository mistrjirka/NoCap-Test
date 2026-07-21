"""Resume-aware wrapper around the existing Rich TTY dashboard."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

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
        if log_path is not None:
            mode = "a" if append_log else "w"
            self._log_console = Console(
                file=log_path.open(mode, encoding="utf-8"), width=999
            )

    def reset_step_timer(self) -> None:
        """Exclude validation, checkpoint I/O, and downtime from TTY step timing."""
        self._last_step_ts = time.perf_counter()
