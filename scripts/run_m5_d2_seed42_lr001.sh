#!/usr/bin/env bash

python train.py \
  --model rowcol_mul_weighted \
  --name M5_Mul_RowCol_d2_lr001 \
  --epochs 100 \
  --seed 42 \
  --gpu-ids 0,1 \
  --depth 2 \
  --conv-ks 5 \
  --local-scale-init 1.0 \
  --rowcol-scale-init 0.1 \
  --lr-max 0.01
