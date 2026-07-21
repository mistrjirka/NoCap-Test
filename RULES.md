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
