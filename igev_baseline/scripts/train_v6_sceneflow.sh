#!/usr/bin/env bash
# ============================================================================
# Training script — IGEV + disparityrender ablation study
# ============================================================================
# Purpose: a clean ablation showing disparityrender > soft-argmin
#
# Experiment A: IGEV original (soft-argmin)
# Experiment B: IGEV + disparityrender
#
# Environment: 2x RTX 5090 (32GB), SceneFlow dataset
# Training time: ~20-24 hours per experiment (200k steps)
#
# Usage:
#   Experiment A (IGEV original):
#     bash scripts/train_v6_sceneflow.sh original
#
#   Experiment B (IGEV + disparityrender):
#     bash scripts/train_v6_sceneflow.sh render
# ============================================================================

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1

# Data path — adjust to your server layout
SCENEFLOW_DIR=${SCENEFLOW_DIR:-"/data/sceneflow"}

MODE=${1:-"render"}

if [ "$MODE" == "original" ]; then
    echo "=== Experiment A: IGEV original (soft-argmin) ==="
    # Use the original igev_stereo_original.py
    cp core/igev_stereo_original.py core/igev_stereo_backup.py
    cp core/igev_stereo_original.py core/igev_stereo.py
    LOGDIR="./checkpoints/v6_igev_original"
elif [ "$MODE" == "render" ]; then
    echo "=== Experiment B: IGEV + disparityrender ==="
    # igev_stereo.py is already the disparityrender version
    # If experiment A was run before, restore the render version
    if [ -f core/igev_stereo_backup.py ]; then
        # Restore the original render version (from git)
        git checkout core/igev_stereo.py
    fi
    LOGDIR="./checkpoints/v6_igev_render"
else
    echo "Usage: bash scripts/train_v6_sceneflow.sh [original|render]"
    exit 1
fi

mkdir -p ${LOGDIR}
echo "Checkpoint dir: ${LOGDIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOGDIR}/train_${TIMESTAMP}.log
echo "Log file: ${LOG_FILE}"

python train_stereo.py \
    --name igev_${MODE} \
    --logdir ${LOGDIR} \
    --batch_size 8 \
    --train_datasets sceneflow \
    --lr 0.0002 \
    --num_steps 200000 \
    --image_size 320 736 \
    --train_iters 22 \
    --valid_iters 32 \
    --mixed_precision \
    --precision_dtype bfloat16 \
    2>&1 | tee -a ${LOG_FILE}
