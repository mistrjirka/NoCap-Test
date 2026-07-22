# LiquidLite Micro-MoE

## Hypothesis

`liquidlite512-moe` keeps LiquidLite's successful hybrid mixer layout and replaces
only four dense MLPs—the third layer in each `shortconv → attention → attention`
group—with parameter-neutral fine-grained MoE layers.

Each MoE layer contains:

```text
1 always-active shared expert, width 384
7 routed experts, width 384 each
top-4 routed experts selected per token
5 active experts per token in total
```

The remaining eight MLPs stay dense. Attention remains ordinary 12-head MHA in
this preset so the MoE effect can be measured separately from GQA4.

## Layer pattern

```text
shortconv + dense MLP
attention + dense MLP
attention + shared/top-4 MoE
```

This three-layer pattern repeats four times. MoE layers are zero-based indices
`2,5,8,11`, corresponding to human-readable layers `3,6,9,12`.

## Parameter accounting

A dense ReLU² MLP has:

```text
2 × 768 × 3072 = 4,718,592 parameters
```

The MoE experts contain the same total hidden width:

```text
1 shared × 384 + 7 routed × 384 = 3072
2 × 768 × 3072 = 4,718,592 expert parameters
```

Each router adds:

```text
768 × 7 = 5,376 parameters
```

Across four MoE layers:

```text
4 × 5,376 = 21,504 additional parameters
LiquidLite:            111,461,888
LiquidLite Micro-MoE:  111,483,392
```

The model size is therefore almost unchanged. The intended gain is conditional
compute, not parameter reduction.

## Active compute

Each token uses one shared and four routed experts:

```text
active hidden width: (1 + 4) × 384 = 1920
original dense width:              3072
```

The causal fixed-capacity implementation reserves 10% routing headroom. In the
ideal balanced case, each MoE layer executes about 67.5% of the dense MLP hidden
compute after including the shared expert and capacity padding. Routing, dispatch,
scatter, and padding overhead can erase this theoretical saving, so RTX 3090/V100
measurement is required.

## Causal routing

Expert slots are allocated independently for each sequence and in token order.
Future tokens cannot evict or change earlier routes, and one sequence cannot
consume another sequence's expert capacity. When an expert exceeds capacity,
later route mass is dropped and the surviving route weights are renormalized.
The shared expert remains active for every token.

This is stricter than selecting the strongest routes over an entire batch, which
would let future tokens change earlier outputs and would not define a valid causal
language model.

## Router objective

Training adds two small router terms:

- load-balancing loss based on selected expert fractions and mean probabilities;
- router z-loss to control logit magnitude.

Validation remains pure next-token cross-entropy. The reported training loss
also remains pure cross-entropy; a zero-value gradient term applies the router
objective without changing its scalar value. The LR controller therefore
observes cross-entropy, not the router auxiliary loss.

## Diagnostics

`moe_test.py` and `moe_cuda_smoke.py` report active experts, dropped route
probability mass, expert-load coefficient of variation, router entropy, and the
training-only auxiliary loss. Run logs keep their existing `train_loss` meaning:
pure next-token cross-entropy. The router objective is attached with a zero-value
gradient term, so it changes gradients without changing the reported scalar loss
or the loss-velocity scheduler input.

## Recommended probes

CPU structure tests:

```bash
python moe_test.py
```

Target-GPU path:

```bash
python moe_cuda_smoke.py
python moe_cuda_smoke.py --compile
```

RTX 3090 adaptive probe:

```bash
bash scripts/run_liquidlite_moe_probe_3090.sh 1536 \
  |& tee liquidlite-moe-probe.log
```

The first PR intentionally fixes routing at 1 shared + top-4 of 7 routed experts
so the architecture is a focused ablation. Test other active-expert ratios in a
follow-up only after this implementation is stable. Keep the MoE experiment only
when equal-step validation remains close and an idle-GPU benchmark shows a real
wall-clock improvement.

## Optional GQA combination

After both ablations work independently, combine them without another preset:

```bash
bash scripts/run_liquidlite_moe_probe_3090.sh 1536 \
  --n_kv_head 4
```

That combined model has 105,191,936 parameters. Do not attribute its result to
MoE alone.
