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
shrink over time and the search stops well before decay.

This is intentionally a conservative AdaLRS-inspired variant without parameter
backtracking. Validation loss never controls LR. The key ablations are:

1. fixed WSD with 1,024-step decay;
2. loss-velocity WSD with the same decay;
3. loss-velocity WSD with a 512-step (~10.7%) decay;
4. each schedule on both `baseline` and `dense512` before combining it with
   LiquidLite.

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
