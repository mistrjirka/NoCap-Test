# LiquidLite-GQA4

## Hypothesis

The completed 1,536-step probes suggest that LiquidLite preserves or slightly
improves equal-step validation loss relative to Dense512. The next conservative
change should reduce only the cost of its eight attention projections while
leaving the successful hybrid layout intact.

`liquidlite512-gqa4` changes:

```text
queries: 12 heads × 64 dimensions
keys:     4 heads × 64 dimensions
values:   4 heads × 64 dimensions
```

Every KV head is shared by three query heads. The four short-convolution layers,
eight global-attention layers, 768-dimensional residual stream, 512-dimensional
tied vocabulary interface, ReLU² MLPs, and all layer positions stay unchanged.

## Parameter accounting

For every attention layer:

```text
ordinary QKV output width: 768 + 768 + 768 = 2304
GQA4 QKV output width:      768 + 256 + 256 = 1280
saving per attention layer: 768 × (2304 - 1280) = 786,432
```

Eight attention layers save:

```text
8 × 786,432 = 6,291,456 parameters
```

Therefore:

```text
LiquidLite-512:       111,461,888
LiquidLite-512-GQA4:  105,170,432
```

## Implementation

CUDA runs call PyTorch scaled-dot-product attention with `enable_gqa=True`.
PyTorch documents GQA as experimental and currently supported by Flash Attention
and the CUDA math backend, with these constraints:

- query-head count must be divisible by KV-head count;
- key and value head counts must match.

CPU tests explicitly repeat KV heads so forward/backward and causality can be
checked without pretending the CPU fallback measures GPU performance.

Existing non-GQA presets retain identical projection shapes and state-dict keys.
Pre-GQA exact-resume checkpoints are normalized as `n_kv_head == n_head`.
Checkpoints cannot be resumed across ordinary LiquidLite and GQA4 because that is
an architecture change.

## Recommended experiments

Architecture-only fixed-schedule comparison:

```bash
bash scripts/run_probe_3090.sh liquidlite512 1536 \
  |& tee liquidlite-fixed-probe.log

bash scripts/run_probe_3090.sh liquidlite512-gqa4 1536 \
  |& tee liquidlite-gqa4-fixed-probe.log
```

Adaptive candidate using the lower peak region selected by the prior LiquidLite
probe:

```bash
bash scripts/run_liquidlite_gqa4_probe_3090.sh 1536 \
  |& tee liquidlite-gqa4-adaptive-probe.log
```

Full candidate:

```bash
bash scripts/run_liquidlite_gqa4_3090.sh \
  |& tee liquidlite-gqa4-full.log
```

V100 equivalents are included. Do not compare timing while gaming or while two
training jobs share one GPU. In that situation, compare equal-step loss only.

## Go/no-go criteria

Keep GQA4 for a full run when, on an otherwise idle GPU:

- validation loss at step 1,536 is no more than about 0.02 worse than LiquidLite;
- median post-compilation step time improves materially;
- no attention backend fallback causes large slowdowns;
- loss remains finite and validation continues to improve.

The 0.02 threshold is an engineering screening heuristic, not a benchmark rule.
