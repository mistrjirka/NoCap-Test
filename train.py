#!/usr/bin/env python3
"""Train baseline and dense NoCap variants on one NVIDIA GPU.

Use torchrun for benchmark runs, even with one GPU, to mirror the public setup:
    torchrun --standalone --nproc_per_node=1 train.py ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import DistributedDataLoader, assert_disjoint_patterns
from model import GPT, config_from_preset
from progress import ProgressDisplay

VAL_TOKENS = 1_048_576
DEFAULT_TARGET = 3.3821


def print0(*args: object, **kwargs: object) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def parse_schedule(value: str) -> list[tuple[int, int]]:
    """Parse `step:length,step:length`; step zero is required."""
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


def sequence_length_at(schedule: list[tuple[int, int]], step: int, default: int) -> int:
    result = default
    for start_step, length in schedule:
        if step < start_step:
            break
        result = length
    return result


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def json_log(path: Path | None, event: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # compatibility with older supported PyTorch versions
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_bin", default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--input_val_bin", default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--output_dir", default="runs")

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
    parser.add_argument(
        "--conv_kernel_size",
        type=int,
        default=None,
        help="Causal depthwise-convolution kernel width",
    )

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
    parser.add_argument(
        "--validation_tokens",
        type=int,
        default=VAL_TOKENS,
        help="Keep 1,048,576 for real benchmark runs; lower only for smoke tests",
    )
    parser.add_argument(
        "--max_initial_val_loss",
        type=float,
        default=20.0,
        help="Abort if step-0 validation is implausibly high; <=0 disables the guard",
    )
    parser.add_argument("--target_val_loss", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--stop_at_target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save_every", type=int, default=5000)

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


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.grad_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if args.sequence_length <= 0 or args.num_iterations <= 0:
        raise ValueError("Sequence length and iterations must be positive")
    if args.validation_tokens <= 0:
        raise ValueError("validation_tokens must be positive")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = 0
        world_size = 1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for benchmark training")
    if args.grad_accumulation_steps % world_size != 0:
        raise ValueError("grad_accumulation_steps must be divisible by world size")
    local_grad_accum = args.grad_accumulation_steps // world_size

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if args.amp == "auto":
        amp_name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        amp_name = args.amp
    if amp_name == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/PyTorch build does not support BF16; use --amp fp16")
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    scaler = make_grad_scaler(enabled=amp_name == "fp16")
    def amp_context():
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    conv_layers_override: tuple[int, ...] | None = None
    if args.conv_layers is not None:
        conv_layers_override = tuple(
            int(item.strip())
            for item in args.conv_layers.split(",")
            if item.strip()
        )

    config = config_from_preset(
        args.preset,
        embedding_dim=args.embedding_dim,
        activation=args.activation,
        embedding_projection=args.embedding_projection,
        qk_norm=args.qk_norm,
        conv_layers=conv_layers_override,
        conv_kernel_size=args.conv_kernel_size,
    )

    assert_disjoint_patterns(args.input_bin, args.input_val_bin)

    # A context schedule preserves tokens per microbatch by changing B inversely with T.
    microbatch_tokens = args.batch_size * args.sequence_length
    val_microbatch_tokens = args.val_batch_size * args.sequence_length
    initial_t = sequence_length_at(args.context_schedule, 0, args.sequence_length)
    if microbatch_tokens % initial_t != 0 or val_microbatch_tokens % initial_t != 0:
        raise ValueError("Scheduled sequence lengths must divide the initial microbatch token counts")
    initial_b = microbatch_tokens // initial_t
    initial_val_b = val_microbatch_tokens // initial_t

    train_loader = DistributedDataLoader(
        args.input_bin, initial_b, initial_t, rank, world_size, device
    )
    val_loader = DistributedDataLoader(
        args.input_val_bin, initial_val_b, initial_t, rank, world_size, device
    )

    model: torch.nn.Module = GPT(config).train().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    init_std = model.tied_weight_std()
    expected_init_std = model.expected_tied_weight_std()
    mixer_summary = model.mixer_summary()
    if not (0.8 * expected_init_std <= init_std <= 1.2 * expected_init_std):
        raise RuntimeError(
            f"Bad tied vocabulary initialization: std={init_std:.6f}, "
            f"expected about {expected_init_std:.6f}. Do not start a full run."
        )
    if hasattr(inductor_config, "coordinate_descent_tuning"):
        inductor_config.coordinate_descent_tuning = True
    if args.compile:
        print0("Compiling model with torch.compile (first call still performs lazy compilation)...")
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = raw_model.configure_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    run_id = str(uuid.uuid4())
    run_dir = Path(args.output_dir) / f"{args.preset}-{run_id}"
    log_path: Path | None = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "events.jsonl"
        metadata = {
            "event": "metadata",
            "run_id": run_id,
            "args": vars(args),
            "model_config": config.to_dict(),
            "parameters": parameter_count,
            "mixer_summary": mixer_summary,
            "tied_weight_init_std": init_std,
            "expected_tied_weight_init_std": expected_init_std,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "torch_version": torch.__version__,
            "amp": amp_name,
            "train_data": train_loader.describe(),
            "validation_data": val_loader.describe(),
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        for source in ("train.py", "model.py", "data_loader.py", "progress.py"):
            shutil.copy2(Path(__file__).with_name(source), run_dir / source)
        json_log(log_path, metadata)

    if args.log_wandb and rank == 0:
        import wandb

        wandb.init(project=args.wandb_project, name=f"{args.preset}-{run_id}")
        wandb.config.update(vars(args))
        wandb.config.update({"model_config": config.to_dict(), "parameters": parameter_count})

    tokens_per_iteration = microbatch_tokens * args.grad_accumulation_steps
    print0(f"GPU: {torch.cuda.get_device_name(device)}; AMP: {amp_name}")
    print0(f"Preset: {args.preset}; parameters: {parameter_count:,}")
    print0(f"Mixer stack: {mixer_summary}")
    print0(
        f"Tied vocabulary init std: {init_std:.6f} "
        f"(expected {expected_init_std:.6f})"
    )
    print0(f"Training loader: {train_loader.describe()}")
    print0(f"Tokens per optimizer step: {tokens_per_iteration:,}")
    if args.context_schedule:
        print0(f"Context schedule: {args.context_schedule}")

    display = ProgressDisplay(
        total_steps=args.num_iterations,
        val_every=args.val_loss_every,
        target_val_loss=args.target_val_loss,
        warmup_iters=args.warmup_iters,
        warmdown_iters=args.warmdown_iters,
        model_params=parameter_count,
        preset=args.preset,
        gpu=torch.cuda.get_device_name(device),
        amp=amp_name,
        tokens_per_step=tokens_per_iteration,
        log_path=str(run_dir / "console.log") if rank == 0 else None,
    )

    def learning_rate_at(step: int) -> float:
        if args.warmup_iters > 0 and step < args.warmup_iters:
            return args.learning_rate * (step + 1) / args.warmup_iters
        if args.warmdown_iters > 0 and step >= args.num_iterations - args.warmdown_iters:
            return args.learning_rate * (args.num_iterations - step) / args.warmdown_iters
        return args.learning_rate

    current_t = initial_t
    x, y = train_loader.next_batch()
    training_time_ms = 0.0
    target_reached = False
    final_step = 0
    torch.cuda.synchronize()
    interval_start = time.perf_counter()

    try:
        for step in range(args.num_iterations + 1):
            final_step = step
            last_step = step == args.num_iterations

            scheduled_t = sequence_length_at(
                args.context_schedule, step, args.sequence_length
            )
            if scheduled_t != current_t:
                if microbatch_tokens % scheduled_t or val_microbatch_tokens % scheduled_t:
                    raise ValueError(
                        f"Sequence length {scheduled_t} does not divide microbatch token counts"
                    )
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.perf_counter() - interval_start)
                current_t = scheduled_t
                new_b = microbatch_tokens // current_t
                new_val_b = val_microbatch_tokens // current_t
                train_loader.set_batch_shape(new_b, current_t)
                val_loader.set_batch_shape(new_val_b, current_t)
                x, y = train_loader.next_batch()
                print0(f"step {step}: switched to B={new_b}, T={current_t}")
                interval_start = time.perf_counter()

            should_validate = (
                args.val_loss_every > 0
                and (step % args.val_loss_every == 0 or last_step)
            )
            if should_validate:
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.perf_counter() - interval_start)
                model.eval()
                val_loader.reset()
                val_b = val_loader.batch_size
                tokens_per_val_step = val_b * current_t * world_size
                if args.validation_tokens % tokens_per_val_step != 0:
                    raise ValueError(
                        f"validation_tokens={args.validation_tokens} is not divisible by "
                        f"validation step size {tokens_per_val_step}"
                    )
                val_steps = args.validation_tokens // tokens_per_val_step
                val_loss = torch.zeros(1, device=device)
                # The official baseline evaluates in FP32, outside autocast.
                with torch.no_grad():
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        assert loss is not None
                        val_loss += loss.detach()
                reduce_mean(val_loss, distributed)
                val_loss /= val_steps
                val_value = float(val_loss.item())
                if (
                    step == 0
                    and args.max_initial_val_loss > 0
                    and val_value > args.max_initial_val_loss
                ):
                    raise RuntimeError(
                        f"Step-0 validation loss {val_value:.4f} exceeds "
                        f"--max_initial_val_loss={args.max_initial_val_loss:.4f}. "
                        "This usually means broken tied-weight initialization."
                    )
                event = {
                    "event": "validation",
                    "step": step,
                    "tokens_seen": step * tokens_per_iteration,
                    "val_loss": val_value,
                    "train_time_s": training_time_ms / 1000.0,
                    "sequence_length": current_t,
                }
                display.update_val(step, val_value)
                if rank == 0:
                    json_log(log_path, event)
                    if args.log_wandb:
                        wandb.log(
                            {
                                "val_loss": val_value,
                                "train_time_s": training_time_ms / 1000.0,
                                "sequence_length": current_t,
                            },
                            step=step * tokens_per_iteration,
                        )

                target_reached = val_value <= args.target_val_loss and step > 0
                if target_reached:
                    print0(
                        f"TARGET REACHED: val loss {val_value:.6f} <= "
                        f"{args.target_val_loss:.4f} after "
                        f"{training_time_ms / 1000.0:.2f}s training time"
                    )
                if target_reached and args.stop_at_target:
                    break
                interval_start = time.perf_counter()

            if last_step:
                break

            model.train()
            train_loss = torch.zeros(1, device=device)
            for micro_step in range(local_grad_accum):
                if isinstance(model, DDP):
                    model.require_backward_grad_sync = micro_step == local_grad_accum - 1
                with amp_context():
                    _, loss = model(x, y, return_logits=False)
                    assert loss is not None
                    train_loss += loss.detach() / local_grad_accum
                    scaled_loss = loss / local_grad_accum
                x, y = train_loader.next_batch()
                scaler.scale(scaled_loss).backward()

            lr = learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            approximate_ms = training_time_ms + 1000.0 * (
                time.perf_counter() - interval_start
            )
            reduce_mean(train_loss, distributed)
            loss_value = float(train_loss.item())
            display.update_train(step, loss_value, lr)
            if rank == 0:
                json_log(
                    log_path,
                    {
                        "event": "train",
                        "step": step,
                        "train_loss": loss_value,
                        "learning_rate": lr,
                        "train_time_s": approximate_ms / 1000.0,
                        "sequence_length": current_t,
                    },
                )

            if rank == 0 and (step + 1) % args.save_every == 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "model_config": config.to_dict(),
                    "training_time_ms": approximate_ms,
                }
                torch.save(checkpoint, run_dir / f"step-{step + 1:06d}.pt")
    finally:
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated(device) // 1024 // 1024
        display.finish(final_step, training_time_ms / 1000.0, peak_mib, target_reached)
        if rank == 0:
            final_checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": final_step,
                "args": vars(args),
                "model_config": config.to_dict(),
                "training_time_ms": training_time_ms,
                "target_reached": target_reached,
            }
            torch.save(final_checkpoint, run_dir / "final.pt")
            json_log(
                log_path,
                {
                    "event": "finished",
                    "step": final_step,
                    "train_time_s": training_time_ms / 1000.0,
                    "target_reached": target_reached,
                    "peak_memory_mib": int(peak_mib),
                },
            )
            if args.log_wandb:
                wandb.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
