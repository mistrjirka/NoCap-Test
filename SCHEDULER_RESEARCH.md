# Scheduler research rationale

## Conclusion for this benchmark

Keep fixed Warmup–Stable–Decay as the control. The most defensible adaptive
experiment is a bounded, training-loss-only search for the stable peak learning
rate, followed by a deterministic terminal decay.

This choice separates two jobs:

- the stable phase uses a sufficiently high LR to make broad optimization
  progress;
- the decay phase suppresses high-LR oscillation and exposes the final loss;
- training-loss descent velocity is used only to tune the plateau height, not to
  decide that the model has "stopped learning".

An ordinary validation-loss or training-loss `ReduceLROnPlateau` scheduler is not
recommended. WSD can intentionally show elevated or slowly changing loss during
the stable phase and then drop rapidly during annealing, so a plateau detector
can trigger too early.

## Why Dense512 is the first target

Dense512 is currently the strongest controlled architecture candidate: its
factorized vocabulary path has a direct step-time rationale and it already has a
corrected training trajectory to compare against. LiquidLite is more novel but
has not yet established equal-token convergence, so scheduler and mixer changes
should not be combined in the first full experiment.

## Evidence used

- AdaLRS motivates online peak-LR search from training-loss descent velocity and
  reports transfer across LLM/VLM pretraining settings and base schedulers.
- MiniCPM introduces WSD and reports that a decay covering about 10% of total
  tokens was sufficient in its tested small-model setting, while 2.5% was not.
- River-valley analyses explain why stable-phase loss may hide useful progress
  that becomes visible only during decay.

The repository implementation is not exact AdaLRS. Exact AdaLRS compares trial
learning rates with model/optimizer backtracking. This speedrun-oriented variant
uses sequential bounded trials instead so checkpoint overhead and token use stay
explicit. It must therefore be judged empirically as a new scheduler variant.

## Recommended ablation matrix

1. Dense512, fixed WSD, 1,024-step decay.
2. Dense512, loss-velocity WSD, 1,024-step decay.
3. Dense512, loss-velocity WSD, 512-step decay.
4. Baseline with whichever scheduler wins steps 1–3.
5. LiquidLite with the winning scheduler only after its fixed-schedule probe is
   competitive in validation loss per wall-clock second.

The official validation shard must never control the adaptive scheduler.
