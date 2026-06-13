#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --mode val \
    --data_dir /path/to/KITTI \
    --dataset_name KITTI_mix \
    --checkpoint_dir checkpoint/kitti \
    --pretrained_vdfnet checkpoint/sceneflow/vdfnet_best.pth \
    --batch_size 3 \
    --val_batch_size 1 \
    --img_height 256 \
    --img_width 512 \
    --val_img_height 384 \
    --val_img_width 1248 \
    --learning_rate 1e-4 \
    --lr_decay_gamma 0.1 \
    --milestones 200,300,350 \
    --max_epoch 400 \
    --save_ckpt_freq 50 \
    --no_validate
