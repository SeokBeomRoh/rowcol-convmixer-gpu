#!/usr/bin/env bash

python train.py --model convmixer   --name M0_ConvMixer_Baseline --epochs 1 --seed 42 --gpu-ids 0,1 --conv-ks 5
python train.py --model rowcol_only --name M1_RowCol_Only       --epochs 1 --seed 42 --gpu-ids 0,1 --conv-ks 5 --rowcol-combine learnable
python train.py --model rowcol_add  --name M2_Additive_RowCol   --epochs 1 --seed 42 --gpu-ids 0,1 --conv-ks 5 --rowcol-combine learnable
