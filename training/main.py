"""Top-level orchestration for the NoCap trainer."""

from __future__ import annotations

from training.cli import build_parser, validate_args
from training.loop import run_training
from training.runtime import (
    build_runtime,
    checkpoint_after_exception,
    finalize_runtime,
)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    runtime = build_runtime(args)
    caught: BaseException | None = None
    try:
        run_training(runtime)
    except BaseException as exc:
        caught = exc
        checkpoint_after_exception(runtime)
        raise
    finally:
        finalize_runtime(runtime, caught)
