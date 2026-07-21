"""Build a fresh or resumed GPU runtime."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import DistributedDataLoader, assert_disjoint_patterns
from model import GPT, config_from_preset
from resume_utils import (
    CHECKPOINT_FORMAT_VERSION,
    load_display_history,
    load_loader_state,
    resolve_resume_path,
    restore_rng_state,
    validate_resume_checkpoint,
)
from training.cli import sequence_length_at
from training.display import ResumableProgressDisplay
from training.runtime import Runtime, json_log, make_grad_scaler, print0


def build_runtime(args: argparse.Namespace) -> Runtime:
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
    if args.resume and world_size != 1:
        raise ValueError("Exact resume currently supports the one-GPU benchmark only")
    local_grad_accum = args.grad_accumulation_steps // world_size

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    amp_name = (
        "bf16"
        if args.amp == "auto" and torch.cuda.is_bf16_supported()
        else "fp16"
        if args.amp == "auto"
        else args.amp
    )
    if amp_name == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/PyTorch build does not support BF16; use --amp fp16")
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    scaler = make_grad_scaler(enabled=amp_name == "fp16")

    conv_layers = None
    if args.conv_layers is not None:
        conv_layers = tuple(
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
        conv_layers=conv_layers,
        conv_kernel_size=args.conv_kernel_size,
    )
    assert_disjoint_patterns(args.input_bin, args.input_val_bin)

    resume_path = resolve_resume_path(args.resume) if args.resume else None
    checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path is not None
        else None
    )

    microbatch_tokens = args.batch_size * args.sequence_length
    val_microbatch_tokens = args.val_batch_size * args.sequence_length
    initial_t = sequence_length_at(args.context_schedule, 0, args.sequence_length)
    if microbatch_tokens % initial_t or val_microbatch_tokens % initial_t:
        raise ValueError("Scheduled lengths must divide the microbatch token counts")
    train_loader = DistributedDataLoader(
        args.input_bin,
        microbatch_tokens // initial_t,
        initial_t,
        rank,
        world_size,
        device,
    )
    val_loader = DistributedDataLoader(
        args.input_val_bin,
        val_microbatch_tokens // initial_t,
        initial_t,
        rank,
        world_size,
        device,
    )

    base_model = GPT(config).train().to(device)
    parameter_count = base_model.parameter_count()
    init_std = base_model.tied_weight_std()
    expected_std = base_model.expected_tied_weight_std()
    if not 0.8 * expected_std <= init_std <= 1.2 * expected_std:
        raise RuntimeError(
            f"Bad tied vocabulary initialization: {init_std:.6f}; "
            f"expected about {expected_std:.6f}"
        )
    optimizer = base_model.configure_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_step = 0
    current_t = initial_t
    training_time_ms = 0.0
    run_id = str(uuid.uuid4())
    wandb_id = uuid.uuid4().hex
    validation_completed_for_step = None
    last_completed_validation_step = None
    if checkpoint is not None:
        validate_resume_checkpoint(
            checkpoint,
            args,
            model_config=config.to_dict(),
            amp_name=amp_name,
            world_size=world_size,
            gpu_name=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
        )
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        load_loader_state(train_loader, checkpoint["train_loader"])
        start_step = int(checkpoint["next_step"])
        current_t = train_loader.sequence_length
        expected_t = sequence_length_at(
            args.context_schedule, start_step, args.sequence_length
        )
        if current_t != expected_t:
            raise ValueError(
                f"Checkpoint uses T={current_t}, but step {start_step} needs T={expected_t}"
            )
        if val_microbatch_tokens % current_t:
            raise ValueError("Restored sequence length does not divide validation tokens")
        val_loader.set_batch_shape(val_microbatch_tokens // current_t, current_t)
        x = checkpoint["prefetched_batch"]["x"].to(device)
        y = checkpoint["prefetched_batch"]["y"].to(device)
        training_time_ms = float(checkpoint["training_time_ms"])
        run_id = str(checkpoint["run_id"])
        wandb_id = str(checkpoint.get("wandb_id", run_id.replace("-", "")))
        completed = checkpoint.get("validation_completed_for_step")
        validation_completed_for_step = int(completed) if completed is not None else None
        last_completed_validation_step = validation_completed_for_step
    else:
        x, y = train_loader.next_batch()

    if hasattr(inductor_config, "coordinate_descent_tuning"):
        inductor_config.coordinate_descent_tuning = True
    model: torch.nn.Module = base_model
    if args.compile:
        print0("Compiling model with torch.compile...")
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    if checkpoint is not None and args.compile and start_step < args.num_iterations:
        print0("Warming resumed compiled graph outside benchmark timing...")
        model.train()
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _, warm_loss = model(x, y, return_logits=False)
            assert warm_loss is not None
        scaler.scale(warm_loss).backward()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
    elif checkpoint is not None:
        restore_rng_state(checkpoint["rng_state"])

    run_dir = (
        resume_path.parent
        if resume_path is not None
        else Path(args.output_dir) / f"{args.preset}-{run_id}"
    )
    log_path = run_dir / "events.jsonl" if rank == 0 else None
    if rank == 0 and checkpoint is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "event": "metadata",
            "run_id": run_id,
            "wandb_id": wandb_id,
            "args": vars(args),
            "model_config": config.to_dict(),
            "parameters": parameter_count,
            "mixer_summary": base_model.mixer_summary(),
            "tied_weight_init_std": init_std,
            "expected_tied_weight_init_std": expected_std,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "torch_version": torch.__version__,
            "amp": amp_name,
            "train_data": train_loader.describe(),
            "validation_data": val_loader.describe(),
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        sources = (
            "train.py",
            "model.py",
            "data_loader.py",
            "progress.py",
            "resume_utils.py",
        )
        for source in sources:
            shutil.copy2(Path(__file__).parents[1] / source, run_dir / source)
        shutil.copytree(
            Path(__file__).parent,
            run_dir / "training",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        json_log(log_path, metadata)
    elif rank == 0:
        json_log(
            log_path,
            {
                "event": "resumed",
                "checkpoint": resume_path.name,
                "next_step": start_step,
                "training_time_s": training_time_ms / 1000.0,
                "timestamp": time.time(),
            },
        )

    wandb_module = None
    if args.log_wandb and rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"{args.preset}-{run_id}",
            id=wandb_id,
            resume="allow",
        )
        wandb_module = wandb
        if checkpoint is None:
            wandb.config.update(vars(args))
            wandb.config.update(
                {"model_config": config.to_dict(), "parameters": parameter_count}
            )

    history_train, history_val = ([], [])
    if checkpoint is not None and log_path is not None:
        history_train, history_val = load_display_history(log_path)
    display = ResumableProgressDisplay(
        total_steps=args.num_iterations,
        val_every=args.val_loss_every,
        target_val_loss=args.target_val_loss,
        warmup_iters=args.warmup_iters,
        warmdown_iters=args.warmdown_iters,
        model_params=parameter_count,
        preset=args.preset,
        gpu=torch.cuda.get_device_name(device),
        amp=amp_name,
        tokens_per_step=microbatch_tokens * args.grad_accumulation_steps,
        log_path=run_dir / "console.log" if rank == 0 else None,
        append_log=checkpoint is not None,
        elapsed_offset_s=training_time_ms / 1000.0,
        initial_train_losses=history_train,
        initial_val_losses=history_val,
    )

    runtime = Runtime(
        args=args,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_grad_accum=local_grad_accum,
        device=device,
        amp_name=amp_name,
        amp_dtype=amp_dtype,
        scaler=scaler,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        base_model=base_model,
        model=model,
        optimizer=optimizer,
        parameter_count=parameter_count,
        mixer_summary=base_model.mixer_summary(),
        microbatch_tokens=microbatch_tokens,
        val_microbatch_tokens=val_microbatch_tokens,
        tokens_per_iteration=microbatch_tokens * args.grad_accumulation_steps,
        start_step=start_step,
        current_t=current_t,
        training_time_ms=training_time_ms,
        run_id=run_id,
        wandb_id=wandb_id,
        validation_completed_for_step=validation_completed_for_step,
        last_completed_validation_step=last_completed_validation_step,
        x=x,
        y=y,
        run_dir=run_dir,
        log_path=log_path,
        display=display,
        checkpoint=checkpoint,
        wandb=wandb_module,
        safe_next_step=start_step,
        target_reached=bool(checkpoint and checkpoint.get("target_reached", False)),
        final_step=start_step,
    )
    runtime.install_signal_handlers()
    return runtime
