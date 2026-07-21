# NoCap Architecture Lab

This is an experimental fork of [BottleCapAI/NoCap-Test](https://github.com/BottleCapAI/NoCap-Test). The original challenge baseline remains available as `train_gpt2.py` and `run.sh`.

Single-GPU experiments for the BottleCapAI NoCap benchmark. The repository keeps
all existing dense designs and adds **LiquidLite-512**, a conservative hybrid of
global causal attention and LFM2-inspired gated short convolutions.

## Presets

| Preset | Layers | Vocabulary interface | MLP | Parameters |
|---|---|---|---|---:|
| `baseline` | 12 attention | tied 768 | GELU | 123,532,032 |
| `dense512` | 12 attention | tied 512 + 512↔768 projections | ReLU² | 111,452,672 |
| `dense512-gated` | 12 attention | gated input projection | ReLU² | 111,845,888 |
| `liquidlite512` | 8 attention + 4 shortconv | tied 512 | ReLU² | 111,461,888 |
| `liquidlite512-gelu` | 8 attention + 4 shortconv | tied 512 | GELU | 111,461,888 |

LiquidLite uses the repeating pattern:

```text
shortconv → attention → attention
shortconv → attention → attention
shortconv → attention → attention
shortconv → attention → attention
```

The short-convolution operator is inspired by Liquid AI's public LFM2 code:

```text
B, C, value = Linear(x).chunk(3)
value = B × value
value = causal depthwise convolution, kernel 3
value = C × value
output = Linear(value)
```

It uses stock PyTorch rather than a custom CUDA extension. This maximizes RTX
3090/V100 portability, but the actual step-time benefit must be measured.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-architecture-lab.txt
```

### RTX 3090

```bash
bash scripts/install_torch_3090.sh
python inspect_env.py
```

The helper installs the official stable PyTorch 2.12.1 CUDA 13.0 wheel. BF16 is used for training. Existing newer builds may also work, but all compared runs must use the same build.

### Tesla V100

```bash
bash scripts/install_torch_v100.sh
python inspect_env.py
```

The V100 uses FP16 with automatic gradient scaling. Do **not** use a CUDA 13
wheel: CUDA 13 removed Volta library and offline-compilation support. The helper
installs a CUDA 12.8 PyTorch wheel.

## Data

One training shard for smoke/probe work:

```bash
python download_data.py --train-shards 1
```

All allowed training data:

```bash
python download_data.py --train-shards 50
```

The trainer rejects overlapping training and validation paths. Full benchmark
runs evaluate exactly 1,048,576 validation tokens.

## Validate the installation

```bash
python smoke_test.py
python resume_test.py
bash scripts/run_smoke.sh
```

The tests check parameter counts, tied-weight initialization, finite
forward/backward passes, strict convolution causality, TTY zoom logic, atomic
checkpoint publication, data-stream continuity, RNG restoration, and AdamW
continuation.

## Recommended experiment order

First run 1,024-step probes rather than another full nine-hour run:

```bash
# RTX 3090
bash scripts/run_probe_3090.sh baseline
bash scripts/run_probe_3090.sh dense512
bash scripts/run_probe_3090.sh liquidlite512
bash scripts/run_probe_3090.sh liquidlite512-gelu

# V100
bash scripts/run_probe_v100.sh baseline
bash scripts/run_probe_v100.sh dense512
bash scripts/run_probe_v100.sh liquidlite512
```

Compare equal-step validation loss, median step time, and validation loss versus
measured training time. Only run the full LiquidLite experiment if it is better
in wall-clock terms or close enough in loss to plausibly recover later.

## Full runs

RTX 3090:

```bash
bash scripts/run_baseline_3090.sh |& tee baseline-3090.log
bash scripts/run_dense_3090.sh |& tee dense512-3090.log
bash scripts/run_liquidlite_3090.sh |& tee liquidlite-3090.log
```

V100:

```bash
bash scripts/run_baseline_v100.sh |& tee baseline-v100.log
bash scripts/run_dense_v100.sh |& tee dense512-v100.log
bash scripts/run_liquidlite_v100.sh |& tee liquidlite-v100.log
```

## Exact resume support

New runs atomically refresh `latest.pt` every 256 completed optimizer steps.
Checkpoint writing and downtime are excluded from the accumulated benchmark
training time. To resume LiquidLite on the RTX 3090, pass either the run
directory or its checkpoint file:

```bash
bash scripts/run_liquidlite_3090.sh \
  --resume runs/liquidlite512-<run-id>

# Equivalent explicit path:
bash scripts/run_liquidlite_3090.sh \
  --resume runs/liquidlite512-<run-id>/latest.pt
```

The same works with the baseline, Dense512, probe, and V100 launchers because
they forward extra arguments to `train.py`.

A single `Ctrl+C` or `SIGTERM` asks the trainer to finish the current optimizer
step and write an exact checkpoint. A second signal aborts immediately. Exact
resume restores:

- model, AdamW, and FP16 GradScaler state;
- the next optimizer-step index and original learning-rate schedule;
- training-shard position and the already-prefetched next batch;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- cumulative measured training time and the original run directory/logs.

Resume is intentionally strict. It rejects changes to the architecture, data
patterns, batch shape, schedule, optimizer settings, validation settings, GPU,
PyTorch version, dtype, or world size. It currently targets the challenge's
one-GPU setup. Old checkpoints created before checkpoint format 2 cannot be
resumed exactly because they lack loader, prefetch, scaler, and RNG state.

Useful controls:

```bash
# Save more often (checkpoint I/O is excluded from benchmark training time):
bash scripts/run_liquidlite_3090.sh --save_every 128

# Disable periodic saves; final.pt is still written at a clean finish:
bash scripts/run_liquidlite_3090.sh --save_every 0

# Keep step-000256.pt, step-000512.pt, ... hard-linked snapshots:
bash scripts/run_liquidlite_3090.sh --keep_step_checkpoints
```

## Better terminal graphs

The TTY now uses explicitly labelled adaptive ranges:

- Training loss shows the most recent 256 steps.
- Validation loss shows the most recent 20 evaluations and always includes the
  challenge target in its range.
- Large early spikes are clipped to robust percentiles instead of flattening the
  useful low-loss region.
- Every graph prints its exact zoom range, so the display does not hide scaling.

Raw values remain unchanged in `events.jsonl` and `console.log`.

## Custom hybrid layouts

Preset layer indices are zero-based. This reproduces LiquidLite manually:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --preset dense512 \
  --conv_layers 0,3,6,9 \
  --conv_kernel_size 3 \
  --amp bf16 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024
```

An empty value disables short convolutions:

```bash
python train.py --preset liquidlite512 --conv_layers '' ...
```

## Fairness

For a valid comparison, keep the GPU, power limit, PyTorch build, dtype, data
order, effective batch size, validation tokens, validation frequency, optimizer,
and learning-rate schedule identical. A different GPU requires rerunning the
baseline on that GPU. Do not train on the validation shard.

## Sources and limitations

- BottleCapAI NoCap-Test provides the benchmark rules, data split, baseline and
  target.
- Liquid AI's LFM2 technical report and public `modeling_lfm2.py` motivate the
  gated short-convolution operator.
- This repository is **not** an official LFM2 implementation. It retains the
  NoCap residual scaling and mostly-global attention to reduce training risk at
  111M parameters and a 2.5B-token budget.
- CUDA convergence and speed must be measured on the target machines; this
  packaging environment has no NVIDIA GPU.
