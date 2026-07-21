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
bash scripts/run_smoke.sh
```

The smoke test checks parameter counts, tied-weight initialization, finite
forward/backward passes, strict convolution causality, and the TTY zoom logic.

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

Never resume checkpoints created before the tied-vocabulary initialization fix.

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
