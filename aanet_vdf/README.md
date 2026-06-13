# AANet backbone + disparityrender

This kit reproduces the **AANet** row of the multi-backbone study. AANet is the
only **2D-CNN** backbone in the set (complementary to the 3D-CNN / iterative
paradigms of GwcNet/PSMNet/IGEV), which makes it the most valuable evidence for
the architecture-agnostic claim. It takes the official
[haofeixu/aanet](https://github.com/haofeixu/aanet) implementation and replaces
*only* its soft-argmin disparity head with `disparityrender`.

## Single changed variable

| | base | render |
|---|---|---|
| Backbone | official AANet (`feature_type aanet` + FPN) | official AANet (same) |
| Disparity head | soft-argmin (`DisparityEstimation`, **untouched**) | disparityrender (patched forward) |
| Data / hyper-params / steps / optimizer / seed | identical | identical |

`disparityrender` matches `igev_baseline/core/submodule.py`, so all four
backbones use the same operator.

## AANet vs GwcNet/PSMNet: a head-shape difference (verified against upstream)

AANet's disparity head is `nets/estimation.py::DisparityEstimation` — **one
shared module called 3 times across scales**, where the cost-volume disparity
axis length **D differs each time** (max_disp//3 per scale: 64 / 32 / 16).
The original forward:
```
cost_volume = cost_volume if self.match_similarity else -cost_volume
prob_volume = F.softmax(cost_volume, dim=1)
disp_candidates = torch.arange(0, D).view(1, D, 1, 1)
disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=False)   # [B, H, W]
```
- A fixed buffer cannot be used (the GwcNet/PSMNet form
  `disparityrender(0, maxdisp-1, maxdisp)` assumes a single D). The patcher
  rewrites the whole forward and **builds `disp_candidates` dynamically**
  (`arange(D-1, -1, -1)`, descending, to match the internal flip).
- Output stays `[B, H, W]` (keepdim=False) — **do not add `.squeeze(1)`**; the
  downstream refinement/loss expects rank-3.
- `forward` returns a list in both train and eval: 3 regressed disparities +
  **2 refinement disparities** = 5 items. The refinement module
  (`StereoDRNetRefinement`) is pure conv with no softmax — it is not a head, so
  the patcher leaves it untouched.
- An **AANet-specific patcher** is therefore used (replaces the whole forward,
  asserts exactly 1 forward matched).

## Files in this directory

| File | Purpose |
|------|---------|
| `disparity_head.py` | the disparityrender module (identical to the IGEV copy, for provenance) |
| `apply_render_patch.py` | AANet-specific patch (adds relu/temperature in `__init__` + rewrites `DisparityEstimation.forward`; touches only `nets/estimation.py`) |
| `train_aanet_sceneflow.sh` | training entry point, run once for `base` and once for `render`, official SceneFlow recipe |
| `eval_aanet_generalization.py` | zero-shot cross-domain evaluation (model construction matches the train script; preprocessing follows predict.py) |
| `README.md` | this file |

## Steps

```bash
cd $VDFNET_ROOT

# 1) clone the official AANet twice
git clone https://github.com/haofeixu/aanet $WORKSPACE/aanet_base
cp -r $WORKSPACE/aanet_base $WORKSPACE/aanet_render

# 1b) AANet requires building deformable conv (official requirement; both copies)
cd $WORKSPACE/aanet_base/nets/deform_conv   && bash build.sh
cd $WORKSPACE/aanet_render/nets/deform_conv && bash build.sh

# 2) patch ONLY the render copy
cd $VDFNET_ROOT
python aanet_vdf/apply_render_patch.py $WORKSPACE/aanet_render
# verify: only nets/estimation.py changed
cd $WORKSPACE/aanet_render && git diff nets/estimation.py
git status   # confirm aanet.py / aggregation.py / refinement.py / cost.py untouched
#   If it reports "matched N != 1", patch the forward by hand using the
#   NEW_FORWARD template in the script.

# 3) train (e.g. 2 GPUs). Official recipe: 64 epochs, MultiStepLR 20,30,40,50,60
cd $VDFNET_ROOT
SCENEFLOW_DIR=$WORKSPACE/SceneFlow bash aanet_vdf/train_aanet_sceneflow.sh base
SCENEFLOW_DIR=$WORKSPACE/SceneFlow bash aanet_vdf/train_aanet_sceneflow.sh render
# best ckpt = aanet_best.pth (selected by val EPE)

# 4) zero-shot cross-domain evaluation (same protocol for both)
cd $WORKSPACE/aanet_base   && python $VDFNET_ROOT/aanet_vdf/eval_aanet_generalization.py \
    --ckpt $VDFNET_CKPT/aanet_base/aanet_best.pth   --tag base
cd $WORKSPACE/aanet_render && python $VDFNET_ROOT/aanet_vdf/eval_aanet_generalization.py \
    --ckpt $VDFNET_CKPT/aanet_render/aanet_best.pth --tag render
```

## Notes / gotchas (verified)

- **deformable conv build**: AANet depends on a custom deform_conv; build it
  with `nets/deform_conv/build.sh` in *both* clones. The build must match your
  PyTorch/CUDA version (the most common AANet environment pitfall).
- **data layout**: AANet uses `--data_dir` pointing at the SceneFlow root (its
  dataloader layout differs slightly from PSMNet's listflowfile). Confirm the
  filename lists under `dataloader/` can locate your `frames_finalpass`.
- **head orientation**: the patched forward uses `arange(D-1, -1, -1)`
  (descending) to match `flip(alpha, [1])`, equivalent to canonical
  disparityrender. Still recommended to overfit a small batch first to confirm
  the disparity orientation.
- **do not add `.squeeze(1)`**: the AANet head outputs `[B, H, W]`
  (keepdim=False); downstream expects rank-3.
- **patch on a fresh official clone**: this repo's `nets/` contains the VDFNet
  model (stereorf), not AANet — always patch a fresh clone of the official aanet
  so the comparison runs against the genuine official backbone.
- **fairness self-check**: `git status` to confirm the patch touches only
  `nets/estimation.py`, with zero changes elsewhere.

## Related

- Same-protocol GwcNet / PSMNet kits: `gwcnet_vdf/`, `psmnet_vdf/`.
