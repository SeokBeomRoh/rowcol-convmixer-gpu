#!/usr/bin/env bash

set -e

for seed in 43 44
do
  echo "============================================================"
  echo "M5 Multiplicative RowCol, depth=2, seed=${seed}, lr=0.01"
  echo "============================================================"
  python train.py \
    --model rowcol_mul_weighted \
    --name M5_Mul_RowCol_d2_lr001 \
    --epochs 100 \
    --seed ${seed} \
    --gpu-ids 0,1 \
    --depth 2 \
    --conv-ks 5 \
    --local-scale-init 1.0 \
    --rowcol-scale-init 0.1 \
    --lr-max 0.01
done
