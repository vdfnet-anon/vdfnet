#!/usr/bin/env bash
# SceneFlow training script (2x RTX 5090 version)

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_DIR=${VDFNET_CKPT:-./checkpoints}/sceneflow_normpe

RESUME_FLAG=""
if ls ${EXP_DIR}/vdfnet_latest.pth 2>/dev/null; then
    echo "Found checkpoint, resuming training..."
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found, starting fresh..."
    mkdir -p ${EXP_DIR}
fi

echo "Experiment dir: ${EXP_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${EXP_DIR}/train_${TIMESTAMP}.log
echo "Log file: ${LOG_FILE}"

torchrun --nproc_per_node=2 train.py \
    --mode train \
    --data_dir ${VDFNET_DATA:-./data/SceneFlow} \
    --dataset_name SceneFlow \
    --batch_size 4 \
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
    --print_freq 50 \
    ${RESUME_FLAG} 2>&1 | tee -a ${LOG_FILE}
