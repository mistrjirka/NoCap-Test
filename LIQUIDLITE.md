# LiquidLite-512 design note

## Hypothesis

Some early and intermediate token mixing may not require full quadratic global
attention. Replace four of twelve attention mixers with inexpensive local gated
causal convolutions while retaining eight global attention layers.

## Architecture

```text
Token IDs
  ↓
512-dimensional tied vocabulary embedding
  ↓ 512→768
[C, A, A] × 4
  ↓ 768→512
Tied vocabulary head
```

`C` is the LFM2-inspired short-convolution mixer. `A` is the unchanged NoCap
causal self-attention mixer. Every layer retains the same 4× MLP and pre-RMSNorm.

## Why four convolution layers

Liquid AI uses a more convolution-heavy mix in larger models trained on vastly
more tokens. NoCap is only about 111M parameters and trains on roughly 2.5B
tokens in the baseline run. Keeping eight global-attention layers is a deliberate
risk reduction for recall and sample efficiency.

## What this experiment can establish

- Whether stock-PyTorch short convolutions actually lower step time on RTX 3090
  and V100.
- Whether replacing four attention layers preserves enough learning efficiency.
- Whether the hybrid improves validation loss per wall-clock second.

It cannot establish that all LFM2 design choices transfer to NoCap. LFM2 also
uses different normalization, GQA, SwiGLU, initialization, data, distillation,
and much longer training.
