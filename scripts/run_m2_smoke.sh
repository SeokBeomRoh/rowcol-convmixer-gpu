#!/usr/bin/env bash

python train.py \
  --model rowcol_add \
  --name M2_Additive_RowCol \
  --epochs 1 \
  --seed 42 \
  --gpu-ids 0,1 \
  --conv-ks 5 \
  --rowcol-combine learnable
