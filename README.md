# RowCol ConvMixer

This project is for experiments on Row/Column-wise convolutional augmentation of ConvMixer.

## Current stage

- M0: ConvMixer baseline on CIFAR-10

## Project structure

```text
rowcol-convmixer/
├── train.py
├── models/
│   └── convmixer.py
├── datasets/
│   └── cifar.py
├── utils/
│   ├── seed.py
│   └── metrics.py
├── configs/
├── scripts/
├── results/
├── checkpoints/
└── data/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# Recommended for RTX 4070 on Linux/WSL.
# This installs CUDA-enabled PyTorch wheels.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages.
pip install -r requirements.txt
```

## GPU check

```bash
python scripts/check_gpu.py
```

Expected output should include:

```text
cuda available: True
gpu count: 2
GPU 0: NVIDIA GeForce RTX 4070 ...
GPU 1: NVIDIA GeForce RTX 4070 ...
```

## Smoke test

```bash
python train.py --model convmixer --epochs 1 --seed 0 --gpu-ids 0,1
```

## Main M0 experiment

```bash
python train.py --model convmixer --epochs 100 --seed 0 --gpu-ids 0,1
python train.py --model convmixer --epochs 100 --seed 1 --gpu-ids 0,1
python train.py --model convmixer --epochs 100 --seed 2 --gpu-ids 0,1
```

## Validation protocol for M3/M5 controls

`bash scripts/run_m3_m5_controls_wsl.sh` runs A–E with `--eval-protocol train_val_test`, `--val-ratio 0.1`, and `--split-seed 42`. Each run uses a class-stratified 90/10 split of the official training set; the official test set is evaluated once after restoring the best validation epoch. Set `SAVE_CHECKPOINT=1` to save that model. Results and checkpoints use new filenames if a matching file already exists.

The default `--eval-protocol legacy` keeps the previous train/test behavior. To run the new protocol directly, use `python train.py --model rowcol_weighted --eval-protocol train_val_test --dataset CIFAR10`; `--dataset CIFAR100` is also supported with the new protocol.

Run the synthetic validation checks with `python scripts/verify_validation_protocol.py` in the configured environment. These checks do not download CIFAR or perform full training.
