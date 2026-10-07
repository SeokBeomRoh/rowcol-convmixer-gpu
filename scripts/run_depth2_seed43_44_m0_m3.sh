#!/usr/bin/env bash

set -e

# Run depth=2 seed extension experiments.
# Common setting:
# - Dataset: CIFAR-10
# - Epochs: 100
# - depth: 2
# - kernel size: 5
# - lr_max: 0.01
# - GPUs: 0,1

echo "============================================================"
echo "Experiment 1/4: M0 ConvMixer, depth=2, seed=43"
echo "============================================================"
python train.py \
  --model convmixer \
  --name M0_ConvMixer_d2_lr001 \
  --epochs 100 \
  --seed 43 \
  --gpu-ids 0,1 \
  --depth 2 \
  --conv-ks 5 \
  --lr-max 0.01

echo "============================================================"
echo "Experiment 2/4: M3 Weighted RowCol, depth=2, seed=43"
echo "============================================================"
python train.py \
  --model rowcol_weighted \
  --name M3_Weighted_RowCol_d2_lr001 \
  --epochs 100 \
  --seed 43 \
  --gpu-ids 0,1 \
  --depth 2 \
  --conv-ks 5 \
  --rowcol-combine learnable \
  --local-scale-init 1.0 \
  --rowcol-scale-init 0.1 \
  --lr-max 0.01

echo "============================================================"
echo "Experiment 3/4: M0 ConvMixer, depth=2, seed=44"
echo "============================================================"
python train.py \
  --model convmixer \
  --name M0_ConvMixer_d2_lr001 \
  --epochs 100 \
  --seed 44 \
  --gpu-ids 0,1 \
  --depth 2 \
  --conv-ks 5 \
  --lr-max 0.01

echo "============================================================"
echo "Experiment 4/4: M3 Weighted RowCol, depth=2, seed=44"
echo "============================================================"
python train.py \
  --model rowcol_weighted \
  --name M3_Weighted_RowCol_d2_lr001 \
  --epochs 100 \
  --seed 44 \
  --gpu-ids 0,1 \
  --depth 2 \
  --conv-ks 5 \
  --rowcol-combine learnable \
  --local-scale-init 1.0 \
  --rowcol-scale-init 0.1 \
  --lr-max 0.01

echo "============================================================"
echo "All depth=2 seed extension experiments completed."
echo "============================================================"
