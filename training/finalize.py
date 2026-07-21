"""Exception checkpointing and runtime cleanup."""

from __future__ import annotations

import torch
import torch.distributed as dist

from training.runtime import Runtime, json_log, print0


def checkpoint_after_exception(runtime: Runtime) -> None:
    if runtime.safe_boundary:
        try:
            runtime.save_checkpoint(
                runtime.safe_next_step,
                reason="exception",
                final=False,
                resume_after=False,
            )
        except Exception as save_exc:
            print0(f"Could not save exception checkpoint: {save_exc}")
    else:
        print0(
            "Failure occurred during a partially accumulated optimizer step; "
            "the previous latest.pt remains the last exact resume point."
        )


def finalize_runtime(
    runtime: Runtime, caught_exception: BaseException | None
) -> None:
    runtime.pause_timer()
    if (
        runtime.safe_boundary
        and caught_exception is None
        and not runtime.final_checkpoint_written
    ):
        reason = "interrupted" if runtime.interrupted else "finished"
        try:
            runtime.save_checkpoint(
                runtime.safe_next_step,
                reason=reason,
                validation_done_step=(
                    runtime.last_completed_validation_step
                    if runtime.last_completed_validation_step
                    == runtime.safe_next_step
                    else None
                ),
                final=True,
                resume_after=False,
            )
        except Exception as save_exc:
            print0(f"Could not save final checkpoint: {save_exc}")

    peak_mib = torch.cuda.max_memory_allocated(runtime.device) // 1024 // 1024
    runtime.display.finish(
        runtime.final_step,
        runtime.training_time_ms / 1000.0,
        peak_mib,
        runtime.target_reached,
    )
    if runtime.rank == 0:
        event_name = (
            "crashed"
            if caught_exception is not None
            else "paused"
            if runtime.interrupted
            else "finished"
        )
        json_log(
            runtime.log_path,
            {
                "event": event_name,
                "step": runtime.final_step,
                "next_step": runtime.safe_next_step,
                "train_time_s": runtime.training_time_ms / 1000.0,
                "target_reached": runtime.target_reached,
                "peak_memory_mib": int(peak_mib),
            },
        )
        if runtime.wandb is not None:
            runtime.wandb.finish()
    runtime.restore_signal_handlers()
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
