# Branch and challenge submission

This repository remains a fork of `BottleCapAI/NoCap-Test`; the upstream Git
history and original baseline files are intentionally preserved.

Development changes are prepared on `agent/liquidlite-architecture-lab`. After
reviewing the smoke-test results and diff, merge that branch into the fork's
`master` branch so a normal clone shows the architecture lab by default.

For a BottleCapAI self-paced submission, record reproducible results in
`IDEA.md` and add `RESULTS.md` after a valid baseline comparison. Then create the
required bundle from a clone containing all branches:

```bash
git fetch --all --prune
git bundle create jirka-svitil.bundle --all
```

Do not commit FineWeb shards, model checkpoints, W&B caches, or run logs.
