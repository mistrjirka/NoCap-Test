# Loss-velocity WSD

## Why this controller exists

The original trainer already uses Warmup–Stable–Decay (WSD): linear warmup,
a long high-LR stable phase, and a terminal linear cooldown. WSD is a strong fit
for NoCap because the target is final pretraining loss under a fixed token/time
budget, and intermediate stable-phase loss can remain artificially elevated
until cooldown.

A normal `ReduceLROnPlateau` rule is a poor match. It can mistake expected WSD
stable-phase behavior or a difficult FineWeb region for convergence and decay too
early. The controller in this repository instead adapts the stable **peak LR**
from training-loss descent velocity, then leaves the final decay deterministic.

## Algorithm

`loss-velocity-wsd` overlays a bounded online LR search on WSD:

1. Warm up linearly to the configured initial peak LR.
2. During `lr_search_start <= step < lr_search_end`, collect training losses in
   fixed windows.
3. Winsorize rare batch spikes, smooth loss, fit `log(loss)` against step by
   least squares, and use the negative slope as relative descent velocity.
4. If velocity falls sufficiently below its EMA, try a mildly higher peak LR.
5. Keep the higher LR only when the next window improves velocity beyond an
   acceptance margin without raising representative loss.
6. A failed higher-LR trial searches slightly below the previous peak.
7. Shrink adjustment magnitudes after every decision.
8. Freeze the selected peak after the search interval and perform a normal
   terminal linear WSD decay.

The implementation is inspired by AdaLRS (arXiv:2506.13274), including online
loss-slope monitoring, early-only search, up-trials, downscaling after failed
trials, and shrinking adjustment factors. It is intentionally more conservative
and does **not** reproduce AdaLRS model/optimizer backtracking. Backtracking would
create large hidden snapshot costs and ambiguous repeated-token accounting in a
speedrun. Therefore, this variant should be evaluated as its own algorithm.

## Recommended NoCap settings

For Dense512 and LiquidLite on RTX 3090/V100:

```text
initial peak LR:       0.0018
search range:          0.00135 .. 0.00220
search steps:          256 .. 3071
velocity window:       128 optimizer steps
upscale factor:        1.08
downscale factor:      1.05
factor decay:          0.90
velocity trigger:      12%
trial acceptance:      3%
loss-rise guard:       6%
terminal decay:        512 steps (candidate)
```

The 512-step decay is approximately 10.7% of a 4,768-step run. MiniCPM's WSD
experiments reported that roughly 10% decay was sufficient in their setting,
but this must be validated on NoCap rather than assumed.

## Experiment order

1. `dense512` fixed WSD, 1,024-step decay.
2. `dense512` loss-velocity WSD, **same 1,024-step decay**.
3. `dense512` loss-velocity WSD, 512-step decay.
4. Repeat the winning scheduler on `baseline` for fairness.
5. Only then combine it with `liquidlite512`.

The 1,536-step probe is useful for stability and LR-decision inspection, but it
cannot prove that a terminal schedule reaches the challenge target.

## TUI and logs

The LR panel displays:

- controller and phase (`warmup`, `lr-search`, `lr-trial`, `stable`, `warmdown`);
- current and selected peak LR;
- loss-window progress;
- latest robust velocity and velocity EMA;
- accepted/rejected decision counts and latest event.

Every training row also records `lr_schedule`, `lr_phase`,
`peak_learning_rate`, and `loss_velocity`. Every LR decision is a separate
`lr_scheduler` event in `events.jsonl`.

## Resume behavior

Exact resume reconstructs adaptive state by replaying logged training losses up
to the checkpoint's `next_step`. Scheduler arguments are checked strictly.
Context-length switches reset slope history and cancel incomplete trials. Old
format-2 checkpoints remain compatible with default fixed WSD.
