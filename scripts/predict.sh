#!/usr/bin/env bash

# KITTI 2015 benchmark submission
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --data_dir /path/to/KITTI2015/testing \
    --pretrained_vdfnet checkpoint/kitti/vdfnet_best.pth \
    --save_type png \
    --visualize

# KITTI 2012 benchmark submission
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --data_dir /path/to/KITTI2012/testing \
#     --pretrained_vdfnet checkpoint/kitti/vdfnet_best.pth \
#     --save_type png

# ETH3D generalization (zero-shot from SceneFlow)
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --data_dir /path/to/ETH3D \
#     --pretrained_vdfnet checkpoint/sceneflow/vdfnet_best.pth \
#     --save_type pfm

# Middlebury generalization (zero-shot from SceneFlow)
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --data_dir /path/to/Middlebury \
#     --pretrained_vdfnet checkpoint/sceneflow/vdfnet_best.pth \
#     --save_type pfm
