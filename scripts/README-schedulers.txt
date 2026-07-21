Scheduler launchers:

- run_loss_velocity_probe_3090.sh: short diagnostic run.
- run_loss_velocity_same_decay_3090.sh: scheduler-only ablation with the original 1,024-step decay.
- run_loss_velocity_3090.sh: scheduler plus 512-step decay candidate.
- run_loss_velocity_same_decay_v100.sh and run_loss_velocity_v100.sh: FP16 V100 equivalents.

All launchers default to Dense512 and accept another preset as their first argument.
