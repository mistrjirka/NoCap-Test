"""Validation and optimizer-step loop for NoCap training."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from training.cli import sequence_length_at
from training.runtime import Runtime, json_log, print0


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def learning_rate_at(runtime: Runtime, step: int) -> float:
    args = runtime.args
    if args.warmup_iters > 0 and step < args.warmup_iters:
        return args.learning_rate * (step + 1) / args.warmup_iters
    if (
        args.warmdown_iters > 0
        and step >= args.num_iterations - args.warmdown_iters
    ):
        return (
            args.learning_rate
            * (args.num_iterations - step)
            / args.warmdown_iters
        )
    return args.learning_rate


def print_startup(runtime: Runtime) -> None:
    print0(f"GPU: {torch.cuda.get_device_name(runtime.device)}; AMP: {runtime.amp_name}")
    print0(f"Preset: {runtime.args.preset}; parameters: {runtime.parameter_count:,}")
    print0(f"Mixer stack: {runtime.mixer_summary}")
    if runtime.checkpoint is not None:
        print0(
            f"Resuming at optimizer step {runtime.start_step}; measured training "
            f"time {runtime.training_time_ms / 1000.0:.2f}s"
        )
    print0(f"Training loader: {runtime.train_loader.describe()}")
    print0(f"Tokens per optimizer step: {runtime.tokens_per_iteration:,}")
    if runtime.args.context_schedule:
        print0(f"Context schedule: {runtime.args.context_schedule}")


def switch_context_if_needed(runtime: Runtime, step: int) -> None:
    scheduled_t = sequence_length_at(
        runtime.args.context_schedule, step, runtime.args.sequence_length
    )
    if scheduled_t == runtime.current_t:
        return

    runtime.pause_timer()
    runtime.safe_boundary = False
    if (
        runtime.microbatch_tokens % scheduled_t
        or runtime.val_microbatch_tokens % scheduled_t
    ):
        raise ValueError(
            f"Sequence length {scheduled_t} does not divide microbatch token counts"
        )
    runtime.current_t = scheduled_t
    train_b = runtime.microbatch_tokens // scheduled_t
    val_b = runtime.val_microbatch_tokens // scheduled_t
    runtime.train_loader.set_batch_shape(train_b, scheduled_t)
    runtime.val_loader.set_batch_shape(val_b, scheduled_t)
    runtime.x, runtime.y = runtime.train_loader.next_batch()
    runtime.safe_boundary = True
    runtime.safe_next_step = step
    print0(f"step {step}: switched to B={train_b}, T={scheduled_t}")
    runtime.display.reset_step_timer()
    runtime.resume_timer()


def validate(runtime: Runtime, step: int, last_step: bool) -> bool:
    skip_restored = runtime.validation_completed_for_step == step
    runtime.validation_completed_for_step = None
    if skip_restored:
        runtime.last_completed_validation_step = step
        return True

    should_validate = runtime.args.val_loss_every > 0 and (
        step % runtime.args.val_loss_every == 0 or last_step
    )
    if not should_validate:
        return False

    runtime.pause_timer()
    runtime.model.eval()
    runtime.val_loader.reset()
    tokens_per_val_step = (
        runtime.val_loader.batch_size * runtime.current_t * runtime.world_size
    )
    if runtime.args.validation_tokens % tokens_per_val_step:
        raise ValueError(
            f"validation_tokens={runtime.args.validation_tokens} is not divisible "
            f"by validation step size {tokens_per_val_step}"
        )
    val_steps = runtime.args.validation_tokens // tokens_per_val_step
    val_loss = torch.zeros(1, device=runtime.device)
    with torch.no_grad():
        for _ in range(val_steps):
            x_val, y_val = runtime.val_loader.next_batch()
            _, loss = runtime.model(x_val, y_val, return_logits=False)
            assert loss is not None
            val_loss += loss.detach()
    reduce_mean(val_loss, runtime.distributed)
    val_loss /= val_steps
    val_value = float(val_loss.item())
    if (
        step == 0
        and runtime.args.max_initial_val_loss > 0
        and val_value > runtime.args.max_initial_val_loss
    ):
        raise RuntimeError(
            f"Step-0 validation loss {val_value:.4f} exceeds "
            f"--max_initial_val_loss={runtime.args.max_initial_val_loss:.4f}"
        )

    runtime.display.update_val(step, val_value)
    json_log(
        runtime.log_path,
        {
            "event": "validation",
            "step": step,
            "tokens_seen": step * runtime.tokens_per_iteration,
            "val_loss": val_value,
            "train_time_s": runtime.training_time_ms / 1000.0,
            "sequence_length": runtime.current_t,
        },
    )
    if runtime.wandb is not None:
        runtime.wandb.log(
            {
                "val_loss": val_value,
                "train_time_s": runtime.training_time_ms / 1000.0,
                "sequence_length": runtime.current_t,
            },
            step=step * runtime.tokens_per_iteration,
        )

    runtime.last_completed_validation_step = step
    runtime.target_reached = (
        val_value <= runtime.args.target_val_loss and step > 0
    )
    if runtime.target_reached:
        print0(
            f"TARGET REACHED: val loss {val_value:.6f} <= "
            f"{runtime.args.target_val_loss:.4f} after "
            f"{runtime.training_time_ms / 1000.0:.2f}s training time"
        )
    runtime.display.reset_step_timer()
    runtime.resume_timer()
    return True


def train_one_step(runtime: Runtime, step: int) -> None:
    runtime.model.train()
    runtime.safe_boundary = False
    train_loss = torch.zeros(1, device=runtime.device)
    for micro_step in range(runtime.local_grad_accum):
        if isinstance(runtime.model, DDP):
            runtime.model.require_backward_grad_sync = (
                micro_step == runtime.local_grad_accum - 1
            )
        with runtime.amp_context():
            _, loss = runtime.model(
                runtime.x, runtime.y, return_logits=False
            )
            assert loss is not None
            train_loss += loss.detach() / runtime.local_grad_accum
            scaled_loss = loss / runtime.local_grad_accum
        runtime.x, runtime.y = runtime.train_loader.next_batch()
        runtime.scaler.scale(scaled_loss).backward()

    lr = learning_rate_at(runtime, step)
    for group in runtime.optimizer.param_groups:
        group["lr"] = lr
    runtime.scaler.step(runtime.optimizer)
    runtime.scaler.update()
    runtime.optimizer.zero_grad(set_to_none=True)
    runtime.safe_boundary = True
    runtime.safe_next_step = step + 1

    measured_ms = runtime.measured_time_ms()
    reduce_mean(train_loss, runtime.distributed)
    loss_value = float(train_loss.item())
    runtime.display.update_train(step, loss_value, lr)
    json_log(
        runtime.log_path,
        {
            "event": "train",
            "step": step,
            "train_loss": loss_value,
            "learning_rate": lr,
            "train_time_s": measured_ms / 1000.0,
            "sequence_length": runtime.current_t,
        },
    )


def run_training(runtime: Runtime) -> None:
    print_startup(runtime)
    runtime.resume_timer()
    for step in range(runtime.start_step, runtime.args.num_iterations + 1):
        runtime.final_step = step
        last_step = step == runtime.args.num_iterations
        switch_context_if_needed(runtime, step)

        validation_done = validate(runtime, step, last_step)
        if runtime.stop_requested:
            runtime.interrupted = True
            runtime.safe_next_step = step
            runtime.save_checkpoint(
                step,
                reason="signal",
                validation_done_step=step if validation_done else None,
                resume_after=False,
            )
            break
        if runtime.target_reached and runtime.args.stop_at_target:
            runtime.safe_next_step = step
            break
        if last_step:
            runtime.safe_next_step = step
            break

        train_one_step(runtime, step)
        if (
            runtime.args.save_every > 0
            and runtime.safe_next_step % runtime.args.save_every == 0
        ):
            runtime.save_checkpoint(runtime.safe_next_step, reason="periodic")
        if runtime.stop_requested:
            runtime.interrupted = True
            runtime.save_checkpoint(
                runtime.safe_next_step,
                reason="signal",
                resume_after=False,
            )
            break
