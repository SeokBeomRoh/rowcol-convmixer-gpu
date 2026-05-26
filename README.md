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
