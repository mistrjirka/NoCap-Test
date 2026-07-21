# Provenance and verification

The dense baseline is a clean-room adaptation of the public MIT-licensed
BottleCapAI/NoCap-Test training program. LiquidLite is independently implemented
from the operator description and public LFM2 model code; this repository does
not copy Transformers framework integration or claim LFM2 compatibility.

## Reference material

- BottleCapAI/NoCap-Test `train_gpt2.py` blob:
  `22960896095ba86f61b53f1c20d89c469fac0ed8`
- BottleCapAI data downloader blob:
  `57b976220338fcff676c7c3c495a4065f13d036d`
- LiquidAI/LFM2-350M public `modeling_lfm2.py`, especially
  `LFM2ShortConv`.
- LFM2 Technical Report, arXiv:2511.23404.

## Local checks completed

- Python bytecode compilation for every Python file.
- `bash -n` for every shell script.
- Exact full-model parameter counts for all presets.
- Correct tied-vocabulary initialization scale.
- Finite forward/backward passes for dense and hybrid tiny models.
- A strict causal-prefix test proving future suffix tokens cannot change earlier
  LiquidLite logits.
- Adaptive TTY zoom-range test.

## Not completed here

- CUDA execution and `torch.compile` performance.
- Full 4,768-step FineWeb convergence.

Those checks require the target RTX 3090 and V100 machines.
