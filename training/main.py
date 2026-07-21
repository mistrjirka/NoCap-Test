"""Top-level orchestration for the NoCap trainer."""

from __future__ import annotations

from training.cli import build_parser, validate_args
from training.finalize import checkpoint_after_exception, finalize_runtime
from training.loop import run_training
from training.lr_scheduler import attach_lr_scheduler
from training.setup import build_runtime


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    runtime = build_runtime(args)
    caught: BaseException | None = None
    scheduler_attached = False
    try:
        attach_lr_scheduler(runtime)
        scheduler_attached = True
        run_training(runtime)
    except BaseException as exc:
        caught = exc
        if scheduler_attached:
            checkpoint_after_exception(runtime)
        raise
    finally:
        finalize_runtime(runtime, caught)
