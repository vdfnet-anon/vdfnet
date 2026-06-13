#!/usr/bin/env bash
# ============================================================================
# eval_kitti_finetuned.sh — evaluate the KITTI fine-tuned VDFNet (IGEV+render)
# ----------------------------------------------------------------------------
# After KITTI fine-tuning (finetune_kitti.sh) finishes, evaluate the fine-tuned
# checkpoint with IGEV's official evaluate_stereo.py to obtain the final KITTI
# D1-all / EPE, and additionally check on ETH3D / Middlebury whether cross-domain
# ability is preserved after fine-tuning.
#
# Usage (from the igev_baseline/ directory):
#   bash $VDFNET_ROOT/igev_baseline/scripts/eval_kitti_finetuned.sh [CKPT]
#   If CKPT is omitted, the latest *.pth under checkpoints/v6_kitti_ft/ is used
#
# Key pitfalls (this machine: RTX 5090 / sm_120):
#   - evaluate_stereo.py already uses nn.DataParallel(..., device_ids=[0]), which
#     is single-GPU safe; we also export CUDA_VISIBLE_DEVICES=0 as a safeguard so
#     the multi-GPU replica path is never triggered.
#   - For the KITTI fine-tuned ckpt, use valid_iters 32 (matching the training valid setting).
# ============================================================================
set -e
cd "$(dirname "$0")/.."   # igev_baseline/

# ---- Select checkpoint: argument takes priority, otherwise latest under v6_kitti_ft ----
CKPT=${1:-}
if [ -z "$CKPT" ]; then
    CKPT=$(ls -t ./checkpoints/v6_kitti_ft/*.pth 2>/dev/null | head -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found. Pass one explicitly:"
    echo "        bash $0 ./checkpoints/v6_kitti_ft/050000_igev_kitti_ft.pth"
    exit 1
fi
echo "[eval] checkpoint: $CKPT"

export CUDA_VISIBLE_DEVICES=0          # single GPU, avoids the sm_120 multi-GPU DataParallel pitfall
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
LOG=./checkpoints/v6_kitti_ft/eval_${TS}.log

# KITTI is the main metric (D1-all); ETH3D/Middlebury check whether cross-domain degrades after fine-tuning.
for DS in kitti eth3d middlebury_H; do
    echo "==================== [$DS] ====================" | tee -a "$LOG"
    python evaluate_stereo.py \
        --restore_ckpt "$CKPT" \
        --dataset "$DS" \
        --valid_iters 32 \
        --mixed_precision \
        2>&1 | tee -a "$LOG"
done

echo ""
echo "=== Evaluation done, log: $LOG ==="
echo "Focus on the [kitti] D1-all / EPE."
