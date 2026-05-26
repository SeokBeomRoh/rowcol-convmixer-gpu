#!/usr/bin/env bash

python train.py \
  --model convmixer \
  --name M0_ConvMixer_Baseline \
  --epochs 1 \
  --seed 0 \
  --gpu-ids 0,1
