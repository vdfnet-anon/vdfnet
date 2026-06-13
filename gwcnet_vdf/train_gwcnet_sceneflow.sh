#!/usr/bin/env bash
# ============================================================================
# train_gwcnet_sceneflow.sh — second-backbone controlled ablation
# ----------------------------------------------------------------------------
# Trains official GwcNet on SceneFlow with one variable changed:
#   base   : original soft-argmin head (clone untouched)
#   render : disparityrender head (apply_render_patch.py applied)
#
# Identical hyper-parameters / steps for both runs. Run base + render
# back-to-back (sequential) or on separate GPUs.
#
# Usage:
#   bash train_gwcnet_sceneflow.sh base
#   bash train_gwcnet_sceneflow.sh render
#
# Prereqs on the GPU server (see README.md for the one-time setup):
#   - GWC_REPO_BASE    = official clone (soft-argmin)
#   - GWC_REPO_RENDER  = patched clone (disparityrender)
#   - SCENEFLOW_DIR    = root with FlyingThings3D/ Monkaa/ Driving/
# ============================================================================
set -e

MODE=${1:-render}
if [ "$MODE" != "base" ] && [ "$MODE" != "render" ]; then
    echo "Usage: bash train_gwcnet_sceneflow.sh [base|render]"; exit 1
fi

# ---- paths (override via env if your layout differs) ----
SCENEFLOW_DIR=${SCENEFLOW_DIR:-${VDFNET_DATA:-./data/SceneFlow}}
GWC_REPO_BASE=${GWC_REPO_BASE:-${WORKSPACE:-.}/GwcNet_base}
GWC_REPO_RENDER=${GWC_REPO_RENDER:-${WORKSPACE:-.}/GwcNet_render}
CKPT_ROOT=${CKPT_ROOT:-${VDFNET_CKPT:-./checkpoints}}

if [ "$MODE" == "base" ]; then
    REPO=$GWC_REPO_BASE;   LOGDIR=$CKPT_ROOT/gwc_base
else
    REPO=$GWC_REPO_RENDER; LOGDIR=$CKPT_ROOT/gwc_render
fi
mkdir -p "$LOGDIR"

# ---- official GwcNet SceneFlow recipe (gwcnet-gc) ----
# 16 epochs, lr 1e-3 (down at epochs 10,12,14,16), batch 8 across 2 GPUs.
# These MUST be identical for base and render — only the head differs.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/train_${TS}.log"
echo "=== GwcNet [$MODE] ==="
echo "repo:    $REPO"
echo "logdir:  $LOGDIR"
echo "log:     $LOG"
echo "data:    $SCENEFLOW_DIR"

# GwcNet's official entry point is main.py. We use its own dataloader
# (filenames lists ship with the repo under ./filenames/sceneflow_*.txt).
python -u main.py \
    --dataset sceneflow \
    --datapath "$SCENEFLOW_DIR" \
    --trainlist ./filenames/sceneflow_train.txt \
    --testlist  ./filenames/sceneflow_test.txt \
    --model gwcnet-gc \
    --maxdisp 192 \
    --epochs 32 \
    --lrepochs "20,24,28,32:2" \
    --batch_size 4 \
    --test_batch_size 2 \
    --lr 0.001 \
    --logdir "$LOGDIR" \
    --summary_freq 100 \
    2>&1 | tee -a "$LOG"

echo "=== done [$MODE]; best checkpoint under $LOGDIR ==="
