#!/usr/bin/env python3
"""Summarize one or more NoCap JSONL logs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_events(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def median_step_seconds(train_events: list[dict[str, object]]) -> float | None:
    if len(train_events) < 3:
        return None
    times = [float(event["train_time_s"]) for event in train_events]
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return None
    return statistics.median(deltas[-256:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--target", type=float, default=3.3821)
    args = parser.parse_args()

    header = (
        f"{'run':34} {'preset':20} {'params':>11} {'best':>8} {'best step':>10} "
        f"{'median ms':>10} {'target time':>12}"
    )
    print(header)
    print("-" * len(header))
    for path in args.logs:
        events = read_events(path)
        metadata = next(event for event in events if event.get("event") == "metadata")
        validations = [event for event in events if event.get("event") == "validation"]
        train_events = [event for event in events if event.get("event") == "train"]
        best_event = min(
            validations,
            key=lambda event: float(event["val_loss"]),
            default=None,
        )
        target = next(
            (
                event
                for event in validations
                if float(event["val_loss"]) <= args.target and int(event["step"]) > 0
            ),
            None,
        )
        best = float(best_event["val_loss"]) if best_event else float("nan")
        best_step = str(best_event["step"]) if best_event else "-"
        target_time = f"{float(target['train_time_s']) / 3600:.3f}h" if target else "not reached"
        median_s = median_step_seconds(train_events)
        median_ms = f"{median_s * 1000:.0f}" if median_s is not None else "-"
        preset = str(metadata.get("args", {}).get("preset", "?"))
        print(
            f"{path.parent.name[:34]:34} {preset[:20]:20} "
            f"{int(metadata['parameters']):11,} {best:8.4f} {best_step:>10} "
            f"{median_ms:>10} {target_time:>12}"
        )


if __name__ == "__main__":
    main()
