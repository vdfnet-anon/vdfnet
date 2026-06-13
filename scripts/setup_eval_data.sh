#!/usr/bin/env bash
# ============================================================================
# setup_eval_data.sh - download & arrange zero-shot eval datasets
# ----------------------------------------------------------------------------
# Downloads ETH3D (two-view training) and Middlebury (MiddEval3 training H) and
# arranges them into the layout expected by evaluate_stereo.py / stereo_datasets.py:
#
#   $EVAL_ROOT/ETH3D/two_view_training/<scene>/{im0.png,im1.png}
#   $EVAL_ROOT/ETH3D/two_view_training_gt/<scene>/disp0GT.pfm
#   $EVAL_ROOT/Middlebury/trainingH/<scene>/{im0.png,im1.png,disp0GT.pfm}
#
# KITTI 2012/2015 require a (free) registered login, so they are NOT auto-
# downloaded here - see the note at the end.
#
# Usage:
#   EVAL_ROOT=/data bash setup_eval_data.sh
#   (defaults to /data; the dataset classes hard-code /data/ETH3D etc., so /data
#    is the simplest choice. If you use another root, symlink it to /data after.)
# ============================================================================
set -e
EVAL_ROOT="${EVAL_ROOT:-/data}"
echo "[setup] EVAL_ROOT=$EVAL_ROOT"
mkdir -p "$EVAL_ROOT"
cd "$EVAL_ROOT"

need() { command -v "$1" >/dev/null 2>&1 || { echo "[error] need '$1' installed"; exit 1; }; }
need wget

# ---------------------------------------------------------------------------
# 1) ETH3D two-view training (images + ground truth), ~70 MB total
# ---------------------------------------------------------------------------
echo "=== [1/2] ETH3D two-view training ==="
mkdir -p "$EVAL_ROOT/ETH3D"
cd "$EVAL_ROOT/ETH3D"
if [ ! -d two_view_training ]; then
    wget -c https://www.eth3d.net/data/two_view_training.7z
    wget -c https://www.eth3d.net/data/two_view_training_gt.7z
    need 7z
    7z x -y two_view_training.7z
    7z x -y two_view_training_gt.7z
    echo "[ok] ETH3D extracted: $(ls two_view_training | wc -l) scenes"
else
    echo "[skip] two_view_training already present"
fi

# ---------------------------------------------------------------------------
# 2) Middlebury MiddEval3 training H (15 scenes, images + GT), ~400 MB
# ---------------------------------------------------------------------------
echo "=== [2/2] Middlebury MiddEval3 training-H ==="
mkdir -p "$EVAL_ROOT/Middlebury"
cd "$EVAL_ROOT/Middlebury"
if [ ! -d trainingH ]; then
    # Official MiddEval3 half-resolution: input images (-data) + GT (-GT0)
    wget -c https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip
    wget -c https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip
    need unzip
    unzip -o -q MiddEval3-data-H.zip
    unzip -o -q MiddEval3-GT0-H.zip
    # Both zips extract to MiddEval3/trainingH/<scene>/...; the dataset class
    # expects $EVAL_ROOT/Middlebury/trainingH/<scene>/, so lift it up one level.
    if [ -d MiddEval3/trainingH ]; then
        cp -rn MiddEval3/trainingH ./trainingH
    fi
    echo "[ok] Middlebury trainingH: $(ls trainingH 2>/dev/null | wc -l) scenes"
else
    echo "[skip] trainingH already present"
fi

# ---------------------------------------------------------------------------
# Summary + KITTI note
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "ETH3D     : $EVAL_ROOT/ETH3D/two_view_training ($(ls "$EVAL_ROOT/ETH3D/two_view_training" 2>/dev/null | wc -l) scenes)"
echo "Middlebury: $EVAL_ROOT/Middlebury/trainingH ($(ls "$EVAL_ROOT/Middlebury/trainingH" 2>/dev/null | wc -l) scenes)"
echo "============================================================"
echo "KITTI is NOT auto-downloaded (requires a free registered login)."
echo "Download KITTI 2012 + 2015 from https://www.cvlibs.net/datasets/kitti/"
echo "and arrange as:"
echo "  $EVAL_ROOT/KITTI/KITTI_2012/training/{colored_0,colored_1,disp_occ}/"
echo "  $EVAL_ROOT/KITTI/KITTI_2015/training/{image_2,image_3,disp_occ_0}/"

