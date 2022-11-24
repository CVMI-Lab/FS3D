#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python tools/train.py \
./configs/prototypical_votenet/train_together_sun/prototypical_votenet_16x8_sunrgbd-3d-10class_1_5.py --sample_num 16 --work_path work_path/sunrgbd_split1_shot5_1

