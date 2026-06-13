#!/usr/bin/env bash
# ============================================================================
# finetune_kitti.sh — fine-tune IGEV+disparityrender on KITTI
# ============================================================================
# Purpose: fine-tune the checkpoint v6_ft4 on the combined KITTI 2012/2015
#          training sets.
#
# Data: KITTI_2015/2012 are under $VDFNET_EVAL_DATA/KITTI/
#       The dataloader in train_stereo.py hardcodes /data/KITTI/, so we align
#       it with a symlink.
#
# Usage (from the igev_baseline/ directory):
#   CUDA_VISIBLE_DEVICES=0 bash ../scripts/finetune_kitti.sh
#   or in the background:
#   CUDA_VISIBLE_DEVICES=0 nohup bash ../scripts/finetune_kitti.sh \
#       > ./kitti_ft.log 2>&1 &
# ============================================================================
set -e
cd "$(dirname "$0")/.."   # igev_baseline/

# ---- 1. Symlink: let the dataloader find the KITTI data ----
KITTI_SRC=${VDFNET_EVAL_DATA:-./data/eval_data}/KITTI
KITTI_DST=/data/KITTI

if [ ! -d "$KITTI_DST/KITTI_2015" ]; then
    echo "[setup] creating symlink $KITTI_DST -> $KITTI_SRC"
    mkdir -p /data
    ln -sfn "$KITTI_SRC" "$KITTI_DST"
else
    echo "[setup] $KITTI_DST already exists, skipping symlink"
fi

# ---- 2. Check the starting checkpoint ----
RESTORE=${RESTORE_CKPT:-"./checkpoints/v6_igev_render_ft4/igev_render_ft4.pth"}
if [ ! -f "$RESTORE" ]; then
    echo "[error] restore checkpoint not found: $RESTORE"
    echo "        set RESTORE_CKPT=<path> to override"
    exit 1
fi
echo "[setup] restoring from: $RESTORE"

# ---- 3. Fine-tuning hyperparameters (IGEV official KITTI fine-tuning protocol) ----
# The official setup uses 50000 steps, lr=1e-4, image_size 320 1152 (wide KITTI images)
# On a single GPU we lower batch to 4 (the original two-GPU setup uses batch=8)
LOGDIR=./checkpoints/v6_kitti_ft
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/train_${TIMESTAMP}.log"
echo "[setup] logdir: $LOGDIR"
echo "[setup] log:    $LOG"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_stereo.py \
    --name igev_kitti_ft \
    --restore_ckpt "$RESTORE" \
    --logdir "$LOGDIR" \
    --train_datasets kitti \
    --lr 0.0001 \
    --num_steps 50000 \
    --batch_size 4 \
    --image_size 320 1152 \
    --train_iters 22 \
    --valid_iters 32 \
    --mixed_precision \
    --precision_dtype bfloat16 \
    2>&1 | tee -a "$LOG"

echo "=== KITTI fine-tuning done, checkpoint in $LOGDIR ==="
echo "Final checkpoint: $LOGDIR/$(ls -t $LOGDIR/*.pth 2>/dev/null | head -1 | xargs basename)"
