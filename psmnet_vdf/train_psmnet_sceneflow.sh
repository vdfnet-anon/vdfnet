#!/usr/bin/env bash
# ============================================================================
# train_psmnet_sceneflow.sh — second-backbone controlled ablation (PSMNet)
# ----------------------------------------------------------------------------
# Trains official PSMNet (JiaRenChang/PSMNet, --model stackhourglass) on
# SceneFlow with ONE variable changed:
#   base   : original soft-argmin heads (clone untouched)
#   render : disparityrender heads (apply_render_patch.py applied)
#
# Identical hyper-parameters for both runs (PSMNet's main.py hardcodes
# Adam lr=1e-3 flat, train batch 12 / test batch 8 — DO NOT edit those for
# one run only). Run base + render back-to-back or on separate GPUs.
#
# Usage:
#   bash train_psmnet_sceneflow.sh base
#   bash train_psmnet_sceneflow.sh render
#
# Prereqs on the GPU server (see README.md):
#   - PSM_REPO_BASE    = official clone (soft-argmin)
#   - PSM_REPO_RENDER  = patched clone (disparityrender)
#   - SCENEFLOW_DIR    = SceneFlow root PSMNet's dataloader expects
#                        (listflowfile.py layout: frames_finalpass + disparity)
# ============================================================================
set -e

MODE=${1:-render}
if [ "$MODE" != "base" ] && [ "$MODE" != "render" ]; then
    echo "Usage: bash train_psmnet_sceneflow.sh [base|render]"; exit 1
fi

# ---- paths (override via env if your layout differs) ----
SCENEFLOW_DIR=${SCENEFLOW_DIR:-${VDFNET_DATA:-./data/SceneFlow}}
PSM_REPO_BASE=${PSM_REPO_BASE:-${WORKSPACE:-.}/PSMNet_base}
PSM_REPO_RENDER=${PSM_REPO_RENDER:-${WORKSPACE:-.}/PSMNet_render}
CKPT_ROOT=${CKPT_ROOT:-${VDFNET_CKPT:-./checkpoints}}
EPOCHS=${EPOCHS:-10}

if [ "$MODE" == "base" ]; then
    REPO=$PSM_REPO_BASE;   SAVE=$CKPT_ROOT/psm_base
else
    REPO=$PSM_REPO_RENDER; SAVE=$CKPT_ROOT/psm_render
fi
mkdir -p "$SAVE"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$SAVE/train_${TS}.log"
echo "=== PSMNet [$MODE] ==="
echo "repo:    $REPO"
echo "save:    $SAVE"
echo "log:     $LOG"
echo "data:    $SCENEFLOW_DIR"

# PSMNet official entry point: main.py (its own SceneFlow dataloader).
# --model stackhourglass is the 3-head model used for all benchmarks.
python -u main.py \
    --maxdisp 192 \
    --model stackhourglass \
    --datapath "$SCENEFLOW_DIR" \
    --epochs "$EPOCHS" \
    --savemodel "$SAVE/" \
    2>&1 | tee -a "$LOG"

echo "=== done [$MODE]; checkpoints (checkpoint_<epoch>.tar) under $SAVE ==="
echo "Pick the epoch with lowest SceneFlow test EPE from the log for eval."
