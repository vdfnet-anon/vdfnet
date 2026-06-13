#!/usr/bin/env bash
# SceneFlow training script (normalized positional encoding version)

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR=~/vdf/checkpoints/sceneflow_normpe_${TIMESTAMP}

echo "Experiment dir: ${EXP_DIR}"

python train.py \
    --mode train \
    --data_dir ~/vdf/data/SceneFlow_hf \
    --dataset_name SceneFlow \
    --batch_size 8 \
    --val_batch_size 1 \
    --img_height 288 \
    --img_width 512 \
    --val_img_height 544 \
    --val_img_width 960 \
    --max_epoch 64 \
    --learning_rate 1e-3 \
    --milestones 20,30,40,50,60 \
    --checkpoint_dir ${EXP_DIR} \
    --save_ckpt_freq 5 \
    --num_workers 8 \
    --print_freq 50
