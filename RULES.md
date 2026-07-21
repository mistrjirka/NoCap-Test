# NoCap rules checklist

The provided full-run scripts are intended to satisfy the public benchmark:

- one GPU process (`--nproc_per_node=1`)
- official FineWeb training shards only
- a separate validation glob rejected if it overlaps training
- no more than the available 5B training tokens
- exactly 1,048,576 validation tokens in full runs
- causal next-token probability model
- target cross-entropy loss 3.3821
- same-hardware baseline required before reporting a speedup

The TTY display is presentation-only. Official comparisons must use raw values
from `events.jsonl`, along with the copied source and `metadata.json`.

Do not use the validation shard for training, data selection, adaptive learning
rate control, routing decisions, or architecture tuning.

Exact resume is valid only from checkpoint format 2 on the same GPU, PyTorch
build, dtype, model/data configuration, optimizer schedule, and one-process
world size. The trainer restores loader position, prefetched input, GradScaler,
RNG, and cumulative measured training time. Checkpoint I/O and machine downtime
are not added to benchmark training time; all resume/checkpoint events remain in
`events.jsonl` for auditability.

Adaptive LR experiments may use training loss only. The official validation
shard remains evaluation-only and must not trigger LR changes. Scheduler
configuration and decisions are written to `events.jsonl`; adaptive state is
reconstructed from the audited per-step training-loss log during exact resume.
When evaluating architectural speedups, report whether the baseline received the
same scheduler and cooldown opportunity.
