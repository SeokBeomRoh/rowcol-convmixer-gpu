# Experiment Log

## 2026-05-27: M0/M1/M2 20-epoch screening on CIFAR-10

Common setting:
- Dataset: CIFAR-10
- Epochs: 20
- Seed: 42
- Hidden dim: 256
- Depth: 8
- Patch size: 2
- Kernel size: 5
- GPUs: RTX 4070 x2
- RowCol combine: learnable

|       Model        | Train Loss | Train Acc | Test Acc | Best Epoch |  Time |
|--------------------|-----------:|----------:|---------:|-----------:|------:|
| M0 ConvMixer       | 0.2518     | 0.9113    | 0.9078   | 19         | 86.6s |
| M1 RowCol-only     | 0.4650     | 0.8369    | 0.8394   | 19         | 27.9s |
| M2 Additive RowCol | 0.2263     | 0.9217    | 0.9118   | 19         | 95.5s |

Observation:
- M1 < M0, suggesting RowCol-only cannot replace local depthwise convolution.
- M2 > M0, suggesting RowCol branch can complement local ConvMixer spatial mixing.
