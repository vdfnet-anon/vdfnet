#!/usr/bin/env bash
# ============================================================================
# check_env.sh — read-only pre-flight check before the multi-backbone /
# KITTI re-submit / recent-method zero-shot experiments.
#
# Does NOT use the GPU for compute, does NOT modify anything. Run on the
# GPU server, paste the FULL output back. It tells us: GPU count/VRAM,
# torch/CUDA, the REAL SceneFlow dir name + layout (PSMNet vs AANet
# dataloaders care), eval_data presence, and whether igev_baseline /
# checkpoints survived the instance release.
#
# Usage:
#   cd <your vdfnet repo>   # the one with gwcnet_vdf/ psmnet_vdf/ aanet_vdf/
#   source env.sh           # sets VDFNET_DATA / VDFNET_EVAL_DATA / VDFNET_CKPT
#   bash check_env.sh 2>&1 | tee check_env.out
# ============================================================================
set +e   # never abort: we want every section's result even if one is missing

# Paths come from the environment (see env.sh.example); fall back to local dirs.
VDFNET_DATA="${VDFNET_DATA:-./data/SceneFlow}"
VDFNET_EVAL_DATA="${VDFNET_EVAL_DATA:-./data/eval_data}"

line() { printf '\n========== %s ==========\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

line "0. REPO & EXPERIMENT DIRS"
echo "cwd: $(pwd)"
echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "HEAD: $(git log --oneline -1 2>/dev/null)"
for d in gwcnet_vdf psmnet_vdf aanet_vdf igev_baseline; do
    if [ -d "$d" ]; then echo "  [OK]  $d/ ($(ls "$d" | wc -l) files)"; else echo "  [!!]  $d/ MISSING"; fi
done

line "1. GPU"
if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
else
    echo "  [!!] nvidia-smi not found"
fi

line "2. PYTHON / TORCH"
python -c "import sys,torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count(), '| bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a')" 2>&1
echo "--- deps the experiments need ---"
python - <<'PY' 2>&1
for m in ["torch","torchvision","numpy","PIL","tqdm","cv2","skimage","tensorboardX","apex"]:
    try:
        mod=__import__(m); v=getattr(mod,"__version__","?")
        print(f"  [OK]  {m} {v}")
    except Exception as e:
        print(f"  [--]  {m}  (not installed: {type(e).__name__})")
PY

line "3. SCENEFLOW (training data) — detect real name + layout"
for root in "$VDFNET_DATA" ./data/SceneFlow; do
    if [ -d "$root" ]; then
        echo "  [OK]  $root  -> top level:"; ls "$root" | sed 's/^/         /' | head
        # probe the canonical FlyingThings/Monkaa/Driving + frames_finalpass + disparity
        for sub in frames_finalpass disparity FlyingThings3D Monkaa Driving TRAIN TEST; do
            found=$(find "$root" -maxdepth 2 -iname "$sub" -type d 2>/dev/null | head -1)
            [ -n "$found" ] && echo "         contains: $found"
        done
    fi
done
# PSMNet dataloader expects a specific layout; AANet uses filename lists — note which exist
echo "  --- PSMNet/AANet filename lists (if any clones already present) ---"
find "${WORKSPACE:-/root}" -maxdepth 3 -iname "sceneflow_train.txt" 2>/dev/null | head

line "4. EVAL DATA (zero-shot) — ETH3D / KITTI / Middlebury"
EVAL="$VDFNET_EVAL_DATA"
for ds in ETH3D KITTI Middlebury; do
    if [ -d "$EVAL/$ds" ]; then echo "  [OK]  $EVAL/$ds"; ls "$EVAL/$ds" | sed 's/^/         /' | head; else echo "  [!!]  $EVAL/$ds MISSING"; fi
done
# spot-check the exact subpaths the eval scripts assume
echo "  --- exact subpaths eval_*_generalization.py assume ---"
for p in \
  "$EVAL/ETH3D/two_view_training" \
  "$EVAL/KITTI/KITTI_2015/training/image_2" \
  "$EVAL/KITTI/KITTI_2015/training/disp_noc_0" \
  "$EVAL/Middlebury/trainingH" ; do
    [ -d "$p" ] && echo "         [OK] $p ($(ls "$p" 2>/dev/null | wc -l) entries)" || echo "         [!!] $p MISSING"
done

line "5. IGEV BASELINE & CHECKPOINTS (KITTI re-submit + recent-method runs)"
echo "--- igev_baseline/ ---"; ls igev_baseline/ 2>/dev/null | sed 's/^/  /' | head -20
echo "--- igev_baseline checkpoints / weights ---"
find igev_baseline -maxdepth 3 \( -name "*.pth" -o -name "*.ckpt" \) 2>/dev/null | sed 's/^/  /' | head
echo "--- repo checkpoints/ ---"; ls -la checkpoints/ 2>/dev/null | sed 's/^/  /'
echo "--- the headline v6_ft4 (0.4686) ckpt present? ---"
find . -path ./.git -prune -o \( -iname "*igev_render_ft4*" -o -iname "*ft4*" \) -print 2>/dev/null | sed 's/^/  /' | head

line "6. DISK"
df -h . "$VDFNET_DATA" 2>/dev/null | sed 's/^/  /'

line "DONE — paste the whole output back"
