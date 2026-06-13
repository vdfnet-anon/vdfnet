#!/usr/bin/env bash
# ============================================================================
# train_aanet_sceneflow.sh — second-backbone controlled ablation (AANet)
# ----------------------------------------------------------------------------
# Trains official AANet (haofeixu/aanet) on SceneFlow with ONE variable
# changed:
#   base   : original soft-argmin head (clone untouched)
#   render : disparityrender head (apply_render_patch.py applied to
#            nets/estimation.py)
#
# Identical hyper-parameters for both runs (official SceneFlow recipe from
# scripts/aanet_train.sh). Only nets/estimation.py's head differs.
#
# Usage:
#   bash train_aanet_sceneflow.sh base
#   bash train_aanet_sceneflow.sh render
#
# Prereqs on the GPU server (see README.md):
#   - AANET_REPO_BASE    = official clone (soft-argmin)
#   - AANET_REPO_RENDER  = patched clone (disparityrender)
#   - SCENEFLOW_DIR      = SceneFlow root AANet expects (data_dir layout)
# ============================================================================
set -e

MODE=${1:-render}
if [ "$MODE" != "base" ] && [ "$MODE" != "render" ]; then
    echo "Usage: bash train_aanet_sceneflow.sh [base|render]"; exit 1
fi

# ---- paths (override via env if your layout differs) ----
SCENEFLOW_DIR=${SCENEFLOW_DIR:-${VDFNET_DATA:-./data/SceneFlow}}
AANET_REPO_BASE=${AANET_REPO_BASE:-${WORKSPACE:-.}/aanet_base}
AANET_REPO_RENDER=${AANET_REPO_RENDER:-${WORKSPACE:-.}/aanet_render}
CKPT_ROOT=${CKPT_ROOT:-${VDFNET_CKPT:-./checkpoints}}
MAX_EPOCH=${MAX_EPOCH:-64}
BATCH=${BATCH:-64}

if [ "$MODE" == "base" ]; then
    REPO=$AANET_REPO_BASE;   SAVE=$CKPT_ROOT/aanet_base
else
    REPO=$AANET_REPO_RENDER; SAVE=$CKPT_ROOT/aanet_render
fi
mkdir -p "$SAVE"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$SAVE/train_${TS}.log"
echo "=== AANet [$MODE] ==="
echo "repo:    $REPO"
echo "save:    $SAVE"
echo "log:     $LOG"
echo "data:    $SCENEFLOW_DIR"

# Official AANet SceneFlow recipe (scripts/aanet_train.sh, first block).
# feature_type aanet + FPN; MultiStepLR milestones 20,30,40,50,60; 64 epochs.
# val_batch_size mirrors batch; val_metric epe selects aanet_best.pth.
python -u train.py \
    --mode val \
    --data_dir "$SCENEFLOW_DIR" \
    --dataset_name SceneFlow \
    --checkpoint_dir "$SAVE" \
    --batch_size "$BATCH" \
    --val_batch_size "$BATCH" \
    --img_height 288 \
    --img_width 576 \
    --val_img_height 576 \
    --val_img_width 960 \
    --feature_type aanet \
    --feature_pyramid_network \
    --max_disp 192 \
    --milestones 20,30,40,50,60 \
    --max_epoch "$MAX_EPOCH" \
    --val_metric epe \
    2>&1 | tee -a "$LOG"

echo "=== done [$MODE]; best ckpt = $SAVE/aanet_best.pth ==="
