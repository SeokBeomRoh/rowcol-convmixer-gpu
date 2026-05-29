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


## 2026-05-28: M0/M1/M2 100-epoch result, seed 42

Common setting:
- Dataset: CIFAR-10
- Epochs: 100
- Seed: 42
- Hidden dim: 256
- Depth: 8
- Patch size: 2
- Kernel size: 5
- GPUs: RTX 4070 x2
- RowCol combine: learnable

| Model | Train Loss | Train Acc | Test Acc | Best Test Acc | Best Epoch | Time |
|---|---:|---:|---:|---:|---:|---:|
| M0 ConvMixer | 0.0861 | 0.9709 | 0.9386 | 0.9389 | 94 | 87.2s |
| M1 RowCol-only | 0.1976 | 0.9322 | 0.8682 | 0.8694 | 97 | 27.8s |
| M2 Additive RowCol | 0.0667 | 0.9775 | 0.9373 | 0.9374 | 98 | 97.6s |

Observation:
- M1 is much lower than M0, confirming that RowCol-only cannot replace local depthwise convolution.
- M2 has higher training accuracy than M0 but slightly lower test accuracy.
- Simple additive RowCol branch may increase training capacity but does not improve generalization in this setting.
- Next step: implement M3 with learnable branch-level scaling.


## 2026-05-28: M3 20-epoch screening, seed 42

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
- M3 local scale init: 1.0
- M3 rowcol scale init: 0.1

| Model | Train Loss | Train Acc | Test Acc | Best Test Acc | Best Epoch | Time |
|---|---:|---:|---:|---:|---:|---:|
| M3 Weighted RowCol | 0.2198 | 0.9229 | 0.9148 | 0.9148 | 19 | 102.5s |

Observation:
- M3 outperforms M0, M1, and M2 in the 20-epoch screening.
- Learnable branch-level scaling appears more stable than simple additive fusion.
- Next step: run M3 for 100 epochs with seed 42 and analyze learned branch scales.

## 2026-05-28: M3 100-epoch instability, seed 42

Setting:
- Model: M3 Weighted RowCol
- Dataset: CIFAR-10
- Epochs: 100
- Seed: 42
- lr_max: 0.03
- local_scale_init: 1.0
- rowcol_scale_init: 0.1

Observation:
- Around epoch 36-40, test accuracy collapsed to nearly random accuracy.
- This collapse coincides with the peak learning rate region.
- Train accuracy remained around 0.83, while test accuracy dropped to 0.10.
- This suggests eval-mode instability, likely related to BatchNorm running statistics and/or unstable branch-scale parameters.

Decision:
- Stop this run.
- Retry M3 with a lower learning rate, starting with lr_max=0.01.

## 2026-05-28: M3 100-epoch result with lower learning rate, seed 42

Common setting:
- Dataset: CIFAR-10
- Epochs: 100
- Seed: 42
- Hidden dim: 256
- Depth: 8
- Patch size: 2
- Kernel size: 5
- GPUs: RTX 4070 x2
- RowCol combine: learnable
- M3 local scale init: 1.0
- M3 rowcol scale init: 0.1
- lr_max: 0.01

| Model | Train Loss | Train Acc | Test Acc | Best Test Acc | Best Epoch | Time |
|---|---:|---:|---:|---:|---:|---:|
| M3 Weighted RowCol | 0.0550 | 0.9814 | 0.9392 | 0.9405 | 94 | 100.1s |

Observation:
- Lowering lr_max from 0.03 to 0.01 stabilizes M3 training.
- M3 achieves 0.9405 best test accuracy, exceeding the M0 baseline result of 0.9389 for seed 42.
- The result supports the hypothesis that RowCol context is useful when its contribution is controlled by learnable branch-level scales.
- Next step: run M0 and M3 for seeds 43 and 44 to estimate mean and standard deviation.

## 2026-05-28: M3 100-epoch result with lower learning rate, seed 42

## Depth ablation: M0 vs M3, depth=2, seed 42, lr_max=0.01

Common setting:
- Dataset: CIFAR-10
- Epochs: 100
- Seed: 42
- Hidden dim: 256
- Depth: 2
- Patch size: 2
- Kernel size: 5
- lr_max: 0.01
- GPUs: RTX 4070 x2

| Model | Depth | Train Loss | Train Acc | Test Acc | Best Test Acc | Best Epoch | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 ConvMixer | 2 | 0.2688 | 0.9074 | 0.8830 | 0.8847 | 92 | 76.3s |
| M3 Weighted RowCol | 2 | 0.2224 | 0.9229 | 0.8891 | 0.8905 | 88 | 80.4s |

Observation:
- M3 outperforms M0 by 0.58 percentage points at depth=2.
- This supports the hypothesis that RowCol branch is more useful in shallow ConvMixer models.
- Next step: run depth=4 comparison.