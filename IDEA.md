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


## Loss-velocity WSD

The scheduler experiment keeps WSD's warmup and terminal linear decay, but uses
only logged training losses to search for a better stable-phase peak learning
rate. It monitors a robust least-squares slope of smoothed log loss. When descent
velocity weakens, it runs a mild higher-LR trial. A clearly better velocity keeps
the trial; a failed trial searches slightly below the prior LR. Adjustment sizes
shrink over time and the search stops well before decay. Validation loss never
controls LR.

## LiquidLite-GQA4

This ablation keeps the complete LiquidLite layout but shares each key/value head
across three query heads in its eight attention layers. The expected advantage is
smaller QKV projections and lower parameter traffic without reducing residual
width, attention depth, MLP capacity, or the number of global mixers.

The model has 105,170,432 parameters, 6,291,456 fewer than LiquidLite. The idea
is successful only if equal-step validation remains close while measured
training time improves on an otherwise idle GPU.

## Required ablations

Compare at least:

1. `baseline`: 768 vocabulary + GELU + 12 attention.
2. `dense512`: 512 vocabulary + ReLU² + 12 attention.
3. `liquidlite512`: 512 vocabulary + ReLU² + 8 attention + 4 shortconv.
4. `liquidlite512-gelu`: isolates the hybrid mixer from ReLU².
5. `liquidlite512-gqa4`: isolates grouped-query attention from other changes.

Use identical hardware, power limit, software, data order and training settings.
Start with 1,024-step probes before full runs.

## Results

| Preset | GPU | Power | PyTorch | AMP | Params | Best val | Step | Train time |
|---|---|---:|---|---|---:|---:|---:|---:|
| baseline | | | | | 123,532,032 | | | |
| dense512 | | | | | 111,452,672 | | | |
| liquidlite512 | | | | | 111,461,888 | | | |
| liquidlite512-gqa4 | | | | | 105,170,432 | | | |
| liquidlite512-gelu | | | | | 111,461,888 | | | |
