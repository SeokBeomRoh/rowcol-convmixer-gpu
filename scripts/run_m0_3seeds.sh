#!/usr/bin/env bash

for seed in 0 1 2
do
  python train.py \
    --model convmixer \
    --name M0_ConvMixer_Baseline \
    --epochs 100 \
    --seed ${seed}
done \
  --gpu-ids 0,1
