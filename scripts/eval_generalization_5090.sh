#!/usr/bin/env bash
# Zero-shot generalization evaluation script (ETH3D + Middlebury H)
# Device: GPU server, 2x RTX 5090
# Usage:   bash scripts/eval_generalization_5090.sh [checkpoint_path]
# Example: bash scripts/eval_generalization_5090.sh $VDFNET_CKPT/sceneflow_normpe/vdfnet_best.pth

set -e

cd "$(dirname "$0")/.."

CKPT=${1:-${VDFNET_CKPT:-./checkpoints}/sceneflow_normpe/vdfnet_best.pth}
DATA_ROOT=${VDFNET_EVAL_DATA:-./data/eval_data}
RESULTS_ROOT=${VDFNET_CKPT:-./checkpoints}/eval_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${RESULTS_ROOT}/eval_${TIMESTAMP}.log

if [ ! -f "$CKPT" ]; then
    echo "Error: checkpoint not found: $CKPT"
    exit 1
fi

mkdir -p $DATA_ROOT $RESULTS_ROOT

echo "=== Zero-shot generalization evaluation ===" | tee -a $LOG_FILE
echo "Checkpoint: $CKPT" | tee -a $LOG_FILE
echo "Timestamp: $TIMESTAMP" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ============================================================
# Step 1: Download and extract datasets
# ============================================================
echo "=== Step 1: Download datasets ===" | tee -a $LOG_FILE

ETH3D_DIR=$DATA_ROOT/ETH3D
if [ ! -d "$ETH3D_DIR/two_view_training" ]; then
    echo "Downloading ETH3D (~70MB)..." | tee -a $LOG_FILE
    mkdir -p $ETH3D_DIR
    unset http_proxy && unset https_proxy
    wget -q --show-progress -P $ETH3D_DIR \
        https://www.eth3d.net/data/two_view_training.7z \
        https://www.eth3d.net/data/two_view_training_gt.7z
    echo "Extracting ETH3D..." | tee -a $LOG_FILE
    pip install py7zr -q
    python -c "
import py7zr, os
os.chdir('$ETH3D_DIR')
py7zr.SevenZipFile('two_view_training.7z').extractall('two_view_training_tmp')
py7zr.SevenZipFile('two_view_training_gt.7z').extractall('two_view_training_tmp')
# Reorganize directory: move scene folders into two_view_training/
import shutil, glob
os.makedirs('two_view_training', exist_ok=True)
for scene in glob.glob('two_view_training_tmp/*'):
    shutil.move(scene, 'two_view_training/')
shutil.rmtree('two_view_training_tmp', ignore_errors=True)
"
    echo "ETH3D ready" | tee -a $LOG_FILE
else
    echo "ETH3D already exists, skipping download" | tee -a $LOG_FILE
fi

MB_DIR=$DATA_ROOT/Middlebury
if [ ! -d "$MB_DIR/MiddEval3/trainingH" ]; then
    echo "Downloading Middlebury H (~200MB)..." | tee -a $LOG_FILE
    mkdir -p $MB_DIR
    unset http_proxy && unset https_proxy
    wget -q --show-progress -P $MB_DIR \
        https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip \
        https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip
    cd $MB_DIR && unzip -q MiddEval3-data-H.zip && unzip -q MiddEval3-GT0-H.zip && cd -
    echo "Middlebury H ready" | tee -a $LOG_FILE
else
    echo "Middlebury H already exists, skipping download" | tee -a $LOG_FILE
fi

echo "" | tee -a $LOG_FILE

# ============================================================
# Step 2: Inference
# ============================================================
echo "=== Step 2: Inference ===" | tee -a $LOG_FILE

# ETH3D: output_dir must contain a two_view_training subdir to match the path expected by eval_eth3d.py
ETH3D_PRED=$RESULTS_ROOT/ETH3D
echo "ETH3D inference..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --data_dir $ETH3D_DIR/two_view_training \
    --pretrained_vdfnet $CKPT \
    --output_dir $ETH3D_PRED/two_view_training \
    --save_type pfm \
    --max_disp 192 2>&1 | tee -a $LOG_FILE
echo "ETH3D inference done" | tee -a $LOG_FILE

# Middlebury H: output_dir must contain a MiddEval3/trainingH subdir
MB_PRED=$RESULTS_ROOT/Middlebury_H
echo "Middlebury H inference..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --data_dir $MB_DIR/MiddEval3/trainingH \
    --pretrained_vdfnet $CKPT \
    --output_dir $MB_PRED/MiddEval3/trainingH \
    --save_type pfm \
    --max_disp 192 2>&1 | tee -a $LOG_FILE
echo "Middlebury H inference done" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE

# ============================================================
# Step 3: Evaluation
# ============================================================
echo "=== Step 3: Evaluation ===" | tee -a $LOG_FILE

echo "--- ETH3D ---" | tee -a $LOG_FILE
python eval_eth3d.py \
    --data_dir $ETH3D_DIR \
    --pred_dir $ETH3D_PRED 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "--- Middlebury H ---" | tee -a $LOG_FILE
python eval_middlebury.py \
    --data_dir $MB_DIR \
    --pred_dir $MB_PRED \
    --resolution H 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "=== Evaluation done, results saved to $LOG_FILE ===" | tee -a $LOG_FILE
