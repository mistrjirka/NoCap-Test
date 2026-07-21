#!/usr/bin/env python3
"""Live terminal progress dashboard for NoCap-Dense training runs."""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_EIGHTH = [" ", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"]


class ProgressDisplay:
    def __init__(
        self,
        total_steps: int,
        val_every: int,
        target_val_loss: float,
        warmup_iters: int,
        warmdown_iters: int,
        model_params: int,
        preset: str,
        gpu: str,
        amp: str,
        tokens_per_step: int,
        log_path: Optional[str] = None,
    ) -> None:
        self._total_steps = total_steps
        self._val_every = val_every
        self._target_val_loss = target_val_loss
        self._warmup_iters = warmup_iters
        self._warmdown_iters = warmdown_iters
        self._model_params = model_params
        self._preset = preset
        self._gpu = gpu
        self._amp = amp
        self._tokens_per_step = tokens_per_step

        self._start_time = time.perf_counter()
        self._last_step_ts = self._start_time
        self._train_losses: List[float] = []
        self._val_losses: List[Tuple[int, float]] = []
        self._best_val: Optional[Tuple[int, float]] = None
        self._step_times: List[float] = []
        self._last_lr: float = 0.0
        self._target_reached: bool = False

        self._is_tty = sys.stdout.isatty()

        self._live: Optional[Live] = None
        if self._is_tty:
            self._live = Live(auto_refresh=False, transient=False)
            self._live.start(refresh=False)

        self._log_console: Optional[Console] = None
        if log_path:
            self._log_console = Console(file=open(log_path, "w", encoding="utf-8"), width=999)

    def update_train(self, step: int, loss: float, lr: float) -> None:
        now = time.perf_counter()
        self._train_losses.append(loss)
        self._last_lr = lr
        delta = now - self._last_step_ts
        self._last_step_ts = now
        self._step_times.append(delta)
        if len(self._step_times) > 100:
            self._step_times.pop(0)

        log_line = self._train_log_line(step, loss, lr)

        if self._live is not None:
            self._render_dashboard(step, lr)

        if self._log_console is not None:
            self._log_console.print(log_line)

        if self._live is None:
            print(log_line, flush=True)

    def update_val(self, step: int, val_loss: float) -> None:
        self._val_losses.append((step, val_loss))
        if self._best_val is None or val_loss < self._best_val[1]:
            self._best_val = (step, val_loss)
        if val_loss <= self._target_val_loss and step > 0:
            self._target_reached = True

        log_line = self._val_log_line(step, val_loss)

        if self._live is not None:
            self._render_dashboard(step, self._last_lr)

        if self._log_console is not None:
            self._log_console.print(log_line)

        if self._live is None:
            print(log_line, flush=True)

        # Exclude validation from the next displayed training-step average.
        # Official benchmark timing still comes from train.py/events.jsonl.
        self._last_step_ts = time.perf_counter()

    def finish(self, step: int, train_time_s: float, peak_mem_mib: int, target_reached: bool) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return

        if self._live is not None:
            self._live.stop()

        final_lines = [
            f"Peak allocated GPU memory: {peak_mem_mib:,} MiB",
            f"Final measured training time: {train_time_s:.2f}s",
        ]
        if target_reached:
            final_lines.append(f"TARGET REACHED: val loss {self._best_val[1]:.6f} <= {self._target_val_loss:.4f}")
        for line in final_lines:
            print(line)

        if self._log_console is not None:
            for line in final_lines:
                self._log_console.print(line)

    # ── log lines (plain text, no ANSI) ────────────────────────────────

    def _train_log_line(self, step: int, loss: float, lr: float) -> str:
        elapsed = self._elapsed()
        avg = self._avg_step_time()
        step_time = avg * 1000 if avg > 0 else 0.0
        eta = avg * (self._total_steps - step) if avg > 0 else 0
        pct = step / max(self._total_steps, 1) * 100
        target_flag = " [TARGET]" if self._target_reached else ""
        return (
            f"step:{step}/{self._total_steps} ({pct:.0f}%) | loss:{loss:.6f} | "
            f"lr:{lr:.6g} | step_time:{step_time:.0f}ms | "
            f"elapsed:{elapsed:.0f}s | eta:{eta:.0f}s{target_flag}"
        )

    def _val_log_line(self, step: int, val_loss: float) -> str:
        elapsed = self._elapsed()
        pct = step / max(self._total_steps, 1) * 100
        best_val = self._best_val[1] if self._best_val else val_loss
        gap = (val_loss - self._target_val_loss) / self._target_val_loss * 100 if self._target_val_loss else 0
        target_flag = " [TARGET REACHED]" if self._target_reached else ""
        return (
            f"step:{step}/{self._total_steps} ({pct:.0f}%) | val:{val_loss:.6f} | "
            f"best_val:{best_val:.6f} | gap:{gap:+.1f}% | elapsed:{elapsed:.0f}s{target_flag}"
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def _avg_step_time(self) -> float:
        if len(self._step_times) < 2:
            return 0.0
        recent = self._step_times[-min(5, len(self._step_times)):]
        return sum(recent) / len(recent)

    @staticmethod
    def _time_str(seconds: float) -> str:
        seconds = max(0.0, seconds)
        if seconds < 60:
            return f"{seconds:.0f}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def _direction_text(self, values: List[float]) -> Text:
        if len(values) < 10:
            return Text("")
        recent = sum(values[-10:]) / 10
        older_start = max(0, len(values) - 20)
        older = sum(values[older_start:len(values) - 10]) / max(1, len(values) - 10 - older_start)
        if older <= 0:
            return Text("")
        if recent < older * 0.995:
            return Text(" (↓)", style="bold green")
        if recent > older * 1.005:
            return Text(" (↑)", style="bold red")
        return Text(" (→)", style="yellow")

    # ── adaptive chart engine ───────────────────────────────────

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        fraction = position - lo
        return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction

    def _zoom_range(
        self,
        values: List[float],
        *,
        target: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Return a robust, explicitly labelled range for low-loss detail.

        The chart clips rare spikes to the 5th/95th percentile instead of letting
        a large early loss flatten the useful late-training region. Validation
        charts always include the challenge target in the visible range.
        """
        if not values:
            return (0.0, 1.0)
        lo = self._quantile(values, 0.05)
        hi = self._quantile(values, 0.95)
        if target is not None:
            lo = min(lo, target)
            hi = max(hi, target)
        span = max(hi - lo, max(abs(hi), 1.0) * 0.015)
        padding = span * 0.10
        return max(0.0, lo - padding), hi + padding

    def _bar_chart(
        self,
        values: List[float],
        graph_w: int,
        graph_h: int,
        *,
        color: str = "cyan",
        value_min: Optional[float] = None,
        value_max: Optional[float] = None,
    ) -> Text:
        if not values or graph_w <= 0 or graph_h <= 0:
            return Text(" " * max(graph_w, 1) * graph_h)

        n = len(values)
        buckets: List[float] = []
        for i in range(graph_w):
            lo = i * n // graph_w
            hi = (i + 1) * n // graph_w
            if lo >= n:
                buckets.append(values[-1])
            else:
                hi = max(hi, lo + 1)
                chunk = values[lo:min(hi, n)]
                # Mean is less misleading than max when many steps share a column.
                buckets.append(sum(chunk) / len(chunk))

        vmin = min(buckets) if value_min is None else value_min
        vmax = max(buckets) if value_max is None else value_max
        if vmax <= vmin:
            vmax = vmin + max(abs(vmin) * 0.01, 0.001)

        max_eighths = graph_h * 8
        grid: List[List[str]] = [[" "] * graph_w for _ in range(graph_h)]

        for col, raw_value in enumerate(buckets):
            value = min(max(raw_value, vmin), vmax)
            eighths = int((value - vmin) / (vmax - vmin) * max_eighths + 0.5)
            eighths = max(0, min(eighths, max_eighths))
            for row in range(graph_h):
                base = row * 8
                if eighths >= base + 8:
                    grid[graph_h - 1 - row][col] = "█"
                elif eighths > base:
                    grid[graph_h - 1 - row][col] = _EIGHTH[eighths - base]

        text = Text()
        for row_idx, row in enumerate(grid):
            text.append("".join(row), style=color)
            if row_idx < len(grid) - 1:
                text.append("\n")
        return text

    # ── dashboard rendering ────────────────────────────────────────────

    def _render_dashboard(self, step: int, lr: float) -> None:
        w = self._terminal_width() - 4
        graph_w = max(w, 20)

        grid = Table.grid(padding=(0, 0))
        grid.add_column(justify="left")

        grid.add_row("")

        grid.add_row(self._progress_row(step))
        grid.add_row("")

        train_view = self._train_losses[-256:]
        train_min, train_max = self._zoom_range(train_view)
        train_chart = self._bar_chart(
            train_view, graph_w, 4, color="cyan",
            value_min=train_min, value_max=train_max,
        )
        last_train = self._train_losses[-1] if self._train_losses else 0.0
        train_dir = self._direction_text(self._train_losses)
        train_heading = Text(" train_loss", style="bold cyan")
        train_heading.append(
            f"  recent {len(train_view)}  zoom {train_min:.3f}–{train_max:.3f}",
            style="bright_black",
        )
        grid.add_row(train_heading)
        grid.add_row(train_chart)
        train_last_line = Text()
        train_last_line.append(f" last: {last_train:.4f}", style="white")
        train_last_line.append_text(train_dir)
        grid.add_row(train_last_line)
        grid.add_row("")

        val_values = [v for _, v in self._val_losses]
        val_view = val_values[-20:]
        val_min, val_max = self._zoom_range(
            val_view, target=self._target_val_loss if val_view else None
        )
        val_chart = self._bar_chart(
            val_view, graph_w, 5, color="magenta",
            value_min=val_min, value_max=val_max,
        )
        last_val = val_values[-1] if val_values else 0.0
        val_dir = self._direction_text(val_values)
        best_val = self._best_val[1] if self._best_val else float("inf")
        best_step = self._best_val[0] if self._best_val else 0
        gap_pct = (last_val - self._target_val_loss) / self._target_val_loss * 100 if self._target_val_loss and val_values else 0

        val_heading = Text(" val_loss", style="bold magenta")
        val_heading.append(
            f"  recent {len(val_view)}  target zoom {val_min:.3f}–{val_max:.3f}",
            style="bright_black",
        )
        grid.add_row(val_heading)
        grid.add_row(val_chart)
        val_last_line = Text()
        val_last_line.append(f" last: {last_val:.4f}", style="white")
        if len(val_values) >= 2:
            delta = last_val - val_values[-2]
            delta_style = "green" if delta < 0 else "red" if delta > 0 else "yellow"
            val_last_line.append(f"  Δ {delta:+.4f}", style=delta_style)
        val_last_line.append_text(val_dir)
        grid.add_row(val_last_line)

        if self._target_reached:
            target_line = Text()
            target_line.append(" TARGET REACHED ", style="bold white on green")
            target_line.append(f"  best: {best_val:.4f} @{best_step}    target: {self._target_val_loss:.4f}")
            grid.add_row(target_line)
        else:
            gap_line = Text()
            gap_line.append(f" best: {best_val:.4f} @{best_step}", style="white")
            gap_line.append(f"    target: {self._target_val_loss:.4f}", style="bright_black")
            if gap_pct < 5:
                gap_style = "bold green"
            elif gap_pct < 20:
                gap_style = "yellow"
            else:
                gap_style = "red"
            gap_line.append(f"    gap: {gap_pct:+.1f}%", style=gap_style)
            grid.add_row(gap_line)
        grid.add_row("")

        grid.add_row(self._lr_row(step, lr))
        grid.add_row("")

        grid.add_row(self._speeds_row())
        grid.add_row("")

        grid.add_row(self._footer_row(step))

        border_style = "bold green" if self._target_reached else "bright_blue"
        panel = Panel(
            grid,
            title=self._title(),
            title_align="left",
            border_style=border_style,
        )

        assert self._live is not None
        self._live.update(panel, refresh=True)

    def _terminal_width(self) -> int:
        try:
            return os.get_terminal_size().columns
        except (ValueError, OSError):
            return 120

    def _title(self) -> str:
        gpu_short = self._gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        params = f"{self._model_params / 1e6:.1f}M" if self._model_params >= 1e6 else str(self._model_params)
        return (
            f"NoCap-Dense  Preset: {self._preset}  GPU: {gpu_short}"
            f"  {self._amp.upper()}  Model: {params}"
        )

    def _progress_row(self, step: int) -> Text:
        pct = step / max(self._total_steps, 1)
        bar_w = max(self._terminal_width() - 30, 20)
        filled = int(pct * bar_w)

        bar_color = "bold green" if self._target_reached else "cyan"
        text = Text()
        text.append("█" * filled, style=bar_color)
        text.append("░" * (bar_w - filled), style="bright_black")
        text.append(f"  {pct * 100:3.0f}%  step: {step}/{self._total_steps}", style="bold")
        return text

    def _lr_row(self, step: int, lr: float) -> Text:
        if self._warmup_iters > 0 and step < self._warmup_iters:
            phase = "warmup"
            pct = (step + 1) / self._warmup_iters
            phase_style = "yellow"
        elif self._warmdown_iters > 0 and step >= self._total_steps - self._warmdown_iters:
            phase = "warmdown"
            steps_in = step - (self._total_steps - self._warmdown_iters)
            pct = 1.0 - steps_in / self._warmdown_iters
            phase_style = "red"
        else:
            phase = "plateau"
            pct = 1.0
            phase_style = "green"

        bar_w = 30
        filled = int(pct * bar_w)

        text = Text()
        text.append("learning rate  ", style="bright_black")
        text.append("█" * filled, style="cyan")
        text.append("░" * (bar_w - filled), style="bright_black cyan")
        text.append(f"  {lr:.6f}  [", style="bright_black")
        text.append(phase, style=phase_style)
        text.append("]", style="bright_black")
        return text

    def _speeds_row(self) -> Text:
        avg = self._avg_step_time()
        tokens_per_sec = self._tokens_per_step / avg if avg > 0 else 0.0
        step_time_ms = avg * 1000 if avg > 0 else 0.0

        try:
            import torch
            gpu_used = torch.cuda.memory_allocated() // (1024 * 1024)
            gpu_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except Exception:
            gpu_used = 0
            gpu_total = 1

        gpu_pct = gpu_used / max(gpu_total, 1) * 100
        gpu_bar_w = 20
        gpu_filled = int(gpu_pct / 100 * gpu_bar_w)

        total_s = self._elapsed()

        text = Text()
        text.append("token_rate  ", style="bright_black")
        text.append(f"{tokens_per_sec:.0f}/s", style="bold cyan")
        text.append("  |  step time  ", style="bright_black")
        text.append(f"{step_time_ms:.0f}ms", style="bold cyan")
        text.append("  |  wall elapsed  ", style="bright_black")
        text.append(f"{total_s:.0f}s", style="cyan")
        text.append("\n")
        text.append("GPU memory  ", style="bright_black")
        text.append(f"{gpu_used} / {gpu_total} MiB", style="white")
        text.append("  ")
        text.append("█" * gpu_filled, style="green" if gpu_pct < 80 else "yellow" if gpu_pct < 95 else "red")
        text.append("░" * (gpu_bar_w - gpu_filled), style="bright_black")
        return text

    def _footer_row(self, step: int) -> Text:
        elapsed = self._elapsed()
        avg = self._avg_step_time()
        remaining = self._total_steps - step
        if avg > 0:
            eta = avg * remaining
            per_100 = avg * 100
        else:
            eta = elapsed / step * remaining if step > 0 else 0
            per_100 = elapsed / step * 100 if step > 0 else 0
        total_est = elapsed + eta

        text = Text()
        text.append("Elapsed  ", style="bright_black")
        text.append(self._time_str(elapsed), style="bold")
        text.append("  |  ETA  ", style="bright_black")
        text.append(self._time_str(eta), style="bold yellow")
        text.append("  |  Total ~", style="bright_black")
        text.append(self._time_str(total_est), style="bold")
        text.append("  |  per 100: ", style="bright_black")
        text.append(f"{per_100:.0f}s", style="cyan")
        return text
