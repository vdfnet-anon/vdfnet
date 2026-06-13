# PSMNet backbone + disparityrender

This kit reproduces the **PSMNet** row of the multi-backbone study: it takes the
official [JiaRenChang/PSMNet](https://github.com/JiaRenChang/PSMNet)
implementation and replaces *only* its soft-argmin disparity head with
`disparityrender`, so that the disparity head is the single changed variable
between the `base` and `render` runs.

## Single changed variable

| | base | render |
|---|---|---|
| Backbone | official PSMNet (`stackhourglass`) | official PSMNet (same) |
| Disparity head | soft-argmin (original 3 heads, **untouched**) | disparityrender (after patch) |
| Data / hyper-params / steps / optimizer / seed | identical | identical |

`disparityrender` is reused **byte-for-byte** across all backbones (see
`disparity_head.py`, identical to the IGEV/GwcNet/AANet copies) — this is what
makes the "one operator, many backbones" argument hold.

## Why patch the official repo (instead of porting into this repo)

- The comparison is most convincing against the *official* `JiaRenChang/PSMNet`
  implementation, with a minimal, auditable patch.
- PSMNet ships its own SceneFlow dataloader (`dataloader/listflowfile.py`) that
  consumes the standard `frames_finalpass + disparity` layout directly.

## PSMNet vs GwcNet: a head-shape difference (verified against upstream)

PSMNet's `models/stackhourglass.py` has **3 disparity heads** (cost1/cost2/cost3):
```
costN = torch.squeeze(costN, 1)
predN = F.softmax(costN, dim=1)
predN = disparityregression(self.maxdisp)(predN)
```
- Training returns `(pred1, pred2, pred3)`; eval returns only `pred3`.
- The cost3 head has a few comment lines between its `softmax` and
  `disparityregression` — the patcher's regex tolerates intervening
  comments/blank lines, so it still matches all 3 heads.
- This differs from GwcNet's functional `disparity_regression(pred, maxdisp)`,
  so a **PSMNet-specific patcher** is used (it asserts `n == 3`).

## Files in this directory

| File | Purpose |
|------|---------|
| `disparity_head.py` | the disparityrender module (identical to the IGEV copy) |
| `apply_render_patch.py` | minimal PSMNet patcher (import + 2 `__init__` lines + replaces the 3 heads; asserts `n == 3`) |
| `train_psmnet_sceneflow.sh` | training entry point, run once for `base` and once for `render`, same hyper-params |
| `eval_psmnet_generalization.py` | zero-shot cross-domain evaluation (same metrics/protocol as the main eval; preprocessing follows PSMNet `submission.py`) |
| `README.md` | this file |

## Steps

```bash
cd $VDFNET_ROOT

# 1) clone the official PSMNet twice
git clone https://github.com/JiaRenChang/PSMNet $WORKSPACE/PSMNet_base
cp -r $WORKSPACE/PSMNet_base $WORKSPACE/PSMNet_render

# 2) patch ONLY the render copy (base stays as the untouched official code)
python psmnet_vdf/apply_render_patch.py $WORKSPACE/PSMNet_render
# verify the diff only touches the disparity head:
cd $WORKSPACE/PSMNet_render && git diff models/
#   expect: 1 import line + 2 __init__ lines + 3 head replacements; feature
#   extraction / hourglass untouched. If it reports "matched N != 3", the
#   upstream head shape changed — patch by hand per disparity_head.py docstring.

# 3) train (e.g. 2 GPUs). Same hyper-params, only the backbone copy differs.
cd $VDFNET_ROOT
SCENEFLOW_DIR=$VDFNET_DATA bash psmnet_vdf/train_psmnet_sceneflow.sh base
SCENEFLOW_DIR=$VDFNET_DATA bash psmnet_vdf/train_psmnet_sceneflow.sh render
# PSMNet main.py defaults to 10 epochs; checkpoints saved as checkpoint_<epoch>.tar

# 4) zero-shot cross-domain evaluation (same protocol for both)
cd $WORKSPACE/PSMNet_base   && python $VDFNET_ROOT/psmnet_vdf/eval_psmnet_generalization.py \
    --ckpt $VDFNET_CKPT/psm_base/checkpoint_9.tar   --tag base
cd $WORKSPACE/PSMNet_render && python $VDFNET_ROOT/psmnet_vdf/eval_psmnet_generalization.py \
    --ckpt $VDFNET_CKPT/psm_render/checkpoint_9.tar --tag render
```

## Notes / gotchas (verified)

- **Head orientation**: `disparityrender` flips the alpha along the disparity
  axis internally (PSMNet cost is in ascending order). Overfit on 1–2 batches
  first to confirm the disparity sign/orientation before a full run.
- **density_temperature** is a new trainable parameter; `optim.Adam(model.parameters())`
  picks it up automatically — no optimizer change needed.
- **Hard-coded lr/bs**: PSMNet `main.py` hard-codes Adam lr=1e-3 (flat) and
  train batch size 12 (not via argparse). `base` and `render` must use the
  **same** main.py config — do not change only one side.
- **Fairness self-check**: before training, `diff` the `main.py` / dataloader of
  the two repo copies to confirm that **only** the disparity head in
  `models/stackhourglass.py` differs.
- **basic variant**: for `--model basic` (single head), passing `--basic` to the
  patcher patches it too (asserts `n == 1`).

## Related

- Same-protocol GwcNet / AANet kits: `gwcnet_vdf/`, `aanet_vdf/`.
