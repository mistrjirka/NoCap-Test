"""Command-line configuration for NoCap training."""

from __future__ import annotations

import argparse

VAL_TOKENS = 1_048_576
DEFAULT_TARGET = 3.3821


def parse_schedule(value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    schedule: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            step_text, length_text = item.split(":", maxsplit=1)
            step, length = int(step_text), int(length_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Schedule must look like 0:256,768:512,2048:1024"
            ) from exc
        if step < 0 or length <= 0:
            raise argparse.ArgumentTypeError("Schedule values must be non-negative")
        schedule.append((step, length))
    schedule.sort()
    if schedule[0][0] != 0:
        raise argparse.ArgumentTypeError("Context schedule must begin at step 0")
    if len({step for step, _ in schedule}) != len(schedule):
        raise argparse.ArgumentTypeError("Context schedule has duplicate steps")
    return schedule


def sequence_length_at(
    schedule: list[tuple[int, int]], step: int, default: int
) -> int:
    result = default
    for start_step, length in schedule:
        if step < start_step:
            break
        result = length
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_bin", default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument(
        "--input_val_bin", default="data/fineweb10B/fineweb_val_*.bin"
    )
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument(
        "--resume",
        default=None,
        help="Exact-format latest.pt/final.pt file, or its run directory",
    )

    parser.add_argument(
        "--preset",
        choices=(
            "baseline",
            "dense512",
            "dense512-gated",
            "liquidlite512",
            "liquidlite512-gelu",
        ),
        default="dense512",
    )
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--activation", choices=("gelu", "relu2"), default=None)
    parser.add_argument(
        "--embedding_projection",
        choices=("linear", "gated-silu"),
        default=None,
    )
    parser.add_argument(
        "--qk_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Experimental Q/K RMS normalization; off in provided presets",
    )
    parser.add_argument(
        "--conv_layers",
        default=None,
        help=(
            "Optional comma-separated zero-based gated-shortconv layer indices; "
            "for example 0,3,6,9. Empty string selects no convolution layers."
        ),
    )
    parser.add_argument("--conv_kernel_size", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accumulation_steps", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument(
        "--context_schedule",
        type=parse_schedule,
        default=[],
        help="Experimental fixed-token schedule, e.g. 0:256,768:512,2048:1024",
    )
    parser.add_argument("--num_iterations", type=int, default=4768)

    parser.add_argument("--learning_rate", type=float, default=0.0018)
    parser.add_argument("--warmup_iters", type=int, default=256)
    parser.add_argument("--warmdown_iters", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--val_loss_every", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--validation_tokens", type=int, default=VAL_TOKENS)
    parser.add_argument("--max_initial_val_loss", type=float, default=20.0)
    parser.add_argument("--target_val_loss", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--stop_at_target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=256,
        help="Atomically refresh latest.pt every N completed steps; 0 disables",
    )
    parser.add_argument(
        "--keep_step_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--amp", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="nocap-dense")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.grad_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if args.sequence_length <= 0 or args.num_iterations <= 0:
        raise ValueError("Sequence length and iterations must be positive")
    if args.validation_tokens <= 0:
        raise ValueError("validation_tokens must be positive")
    if args.save_every < 0:
        raise ValueError("save_every cannot be negative")
