# Architecture experiments

## Dense512

Dense512 keeps the public 12-layer, 768-wide Transformer core but factorizes the
tied vocabulary interface to 512 dimensions and uses ReLU². The vocabulary
factorization targets a large per-token matrix multiplication while retaining the
core width.

## LiquidLite-512

LiquidLite keeps the Dense512 vocabulary interface and MLP, then replaces layers
1, 4, 7 and 10's attention mixers with LFM2-inspired gated causal depthwise
convolutions. Eight global attention layers remain unchanged.

The hypothesis is that four local mixers lower wall-clock step time without
meaningfully damaging convergence. The experiment is successful only if it
improves validation loss per measured training second—not merely FLOPs or
parameter count.

## Required ablations

Compare at least:

1. `baseline`: 768 vocabulary + GELU + 12 attention.
2. `dense512`: 512 vocabulary + ReLU² + 12 attention.
3. `liquidlite512`: 512 vocabulary + ReLU² + 8 attention + 4 shortconv.
4. `liquidlite512-gelu`: isolates the hybrid mixer from ReLU².

Use identical hardware, power limit, software, data order and training settings.
Start with 1,024-step probes before full runs.

## Results

| Preset | GPU | Power | PyTorch | AMP | Params | Best val | Step | Train time |
|---|---|---:|---|---|---:|---:|---:|---:|
| baseline | | | | | 123,532,032 | | | |
| dense512 | | | | | 111,452,672 | | | |
| liquidlite512 | | | | | 111,461,888 | | | |
| liquidlite512-gelu | | | | | 111,461,888 | | | |
