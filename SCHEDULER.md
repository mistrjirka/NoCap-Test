# Monotonic WSqD schedulers

The completed 1,536-step GQA4 and Micro-MoE probes showed the same failure mode:
the forward-only `loss-velocity-wsd` controller accepted higher learning rates
late, while most visible loss improvement arrived during the short final
cooldown.

That controller is retained for reproducibility. It is not a faithful AdaLRS
implementation: AdaLRS (arXiv:2506.13274) snapshots and backtracks model and
optimizer state around LR trials, and its ablation reports that removing
backtracking can leave lasting damage after an oversized trial. Hidden snapshot
cost, repeated tokens, and timing ambiguity are poor fits for this speedrun.

## `wsqd`

`wsqd` implements a shifted inverse-square-root base followed by a guaranteed
linear cooldown, inspired by WSqD (arXiv:2607.10959):

```text
linear warmup
    ↓
shifted inverse-square-root base
    ↓
reserved final linear cooldown
```

After warmup:

```text
base_lr = peak_lr × sqrt((T0 + 1) / (T0 + post_warmup_step))
```

The first base step equals the configured peak. The base rate then decreases
continuously; there is no flat high-LR middle phase. Power Scheduler
(arXiv:2408.13359) independently finds an approximately inverse-square-root
relationship between useful LR and token horizon.

## `loss-aware-wsqd`

This experimental overlay uses **training loss only** and can only lower LR:

```text
effective_lr = wsqd_base_lr × monotonic_multiplier
```

For every complete robust loss window it:

1. clips rare batch-loss spikes;
2. smooths loss and fits log-loss velocity;
3. compares median loss with the preceding window;
4. requires both low median improvement and weak velocity;
5. requires multiple consecutive stalled windows;
6. multiplies LR downward;
7. waits through a cooldown before another decision.

The multiplier begins at `1.0`, is bounded below, and never increases.
Validation loss never controls LR.

The supplied probes use:

```text
window:                     128 steps
minimum median improvement: 2.25%
confirmation:               2 windows
downward factor:            0.80
cooldown:                   2 windows
minimum multiplier:         0.40
final cooldown:             about 20% of the probe
```

A replay of the uploaded GQA-like loss trend produces a downward decision and
cannot increase LR. This validates controller direction, not counterfactual
model quality; only a new GPU run can establish final loss.

## 1,536-step tests

Paper-backed WSqD controls:

```bash
bash scripts/run_wsqd_probe_3090.sh liquidlite512-gqa4 1536
bash scripts/run_wsqd_probe_3090.sh liquidlite512-moe 1536
```

Loss-aware variants:

```bash
bash scripts/run_loss_aware_wsqd_probe_3090.sh liquidlite512-gqa4 1536
bash scripts/run_loss_aware_wsqd_probe_3090.sh liquidlite512-moe 1536
```

Launchers forward extra arguments, for example:

```bash
bash scripts/run_loss_aware_wsqd_probe_3090.sh liquidlite512-moe 1536 \
  --learning_rate 0.00155
```

## Full runs

```bash
bash scripts/run_wsqd_3090.sh liquidlite512-gqa4
bash scripts/run_loss_aware_wsqd_3090.sh liquidlite512-gqa4
```

Equivalent V100 FP16 launchers are included.

## TUI and logs

The TUI reports current LR, WSqD base LR, monotonic multiplier, loss-window
improvement, velocity, plateau streak, decisions, and the latest LR event.

Training events add:

```text
base_learning_rate
lr_multiplier
loss_window_improvement
```

Exact adaptive resume reconstructs state by replaying `events.jsonl`. All new
scheduler arguments are checked strictly. Pure `wsqd` is deterministic and
does not need loss replay.
