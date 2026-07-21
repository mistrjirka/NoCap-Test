# Scheduler experiment results

Record each run before changing controller parameters.

| Architecture | Scheduler | Warmdown | GPU | Power | Peak LR selected | Target time | Best val | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|
| Dense512 | fixed WSD | 1024 | | | 0.0018 | | | control |
| Dense512 | loss-velocity WSD | 1024 | | | | | | scheduler-only ablation |
| Dense512 | loss-velocity WSD | 512 | | | | | | shorter-decay candidate |
| Baseline | winning schedule | | | | | | | fairness control |
| LiquidLite512 | winning schedule | | | | | | | only after fixed-schedule probe |

For adaptive runs, copy all `lr_scheduler` events from `events.jsonl` or attach
the full log. Report the same PyTorch build, dtype, power limit, data order,
effective batch size, and validation settings for every compared row.
