# GwcNet backbone + disparityrender

This kit reproduces the **GwcNet** row of the multi-backbone study: it takes the
official [xy-guo/GwcNet](https://github.com/xy-guo/GwcNet) implementation and
replaces *only* its soft-argmin disparity head with `disparityrender`, so the
disparity head is the single changed variable between `base` and `render`.

## Single changed variable

The value of a controlled comparison rests on **fairness**: the only difference
between the two runs may be the disparity head.

| | base | render |
|---|---|---|
| Backbone | official GwcNet (`gwcnet-gc`) | official GwcNet (same) |
| Disparity head | soft-argmin (original, **untouched**) | disparityrender (after patch) |
| Data / hyper-params / steps / optimizer / seed | identical | identical |

`disparityrender` is reused **byte-for-byte** across backbones (see the header
of `disparity_head.py`), so IGEV and GwcNet use the **same operator** — the
premise of the "one operator, multiple backbones" argument.

## Why patch the official repo (instead of porting into this repo's train.py)

- The comparison is most convincing against the *official* `xy-guo/GwcNet`
  implementation; rewriting it into another framework would raise doubts about
  whether it is still GwcNet.
- GwcNet's official dataloader consumes the standard
  `FlyingThings3D/Monkaa/Driving` (`frames_finalpass`) layout directly, so no
  data conversion is needed.

## Files in this directory

| File | Purpose |
|------|---------|
| `disparity_head.py` | the disparityrender module (identical to the IGEV copy) + soft-argmin fallback |
| `apply_render_patch.py` | minimal patch for an official GwcNet clone (head + 3 init lines only) |
| `train_gwcnet_sceneflow.sh` | training entry point, run once for `base` and once for `render`, same hyper-params |
| `eval_gwcnet_generalization.py` | zero-shot cross-domain evaluation (same metric definitions as the main eval) |
| `README.md` | this file |

## Steps

```bash
# 0) enter the project root (contains this gwcnet_vdf/ directory)
cd $VDFNET_ROOT

# 1) clone the official GwcNet twice: base (untouched) + render (patched)
git clone https://github.com/xy-guo/GwcNet $WORKSPACE/GwcNet_base
cp -r $WORKSPACE/GwcNet_base $WORKSPACE/GwcNet_render

# 2) patch ONLY the render copy (base stays as the untouched official code)
python gwcnet_vdf/apply_render_patch.py $WORKSPACE/GwcNet_render
# verify the patch only touches the disparity head:
cd $WORKSPACE/GwcNet_render && git diff models/gwcnet.py
#   expect: 1 import line + 3 __init__ lines (render/temperature) + head swap.
#   If it reports "no soft-argmin heads matched", the upstream head shape
#   differs — patch by hand per the disparity_head.py docstring (head lines only).

# 3) train (e.g. 2 GPUs). Same hyper-params, only the backbone copy differs.
cd $VDFNET_ROOT
SCENEFLOW_DIR=$VDFNET_DATA bash gwcnet_vdf/train_gwcnet_sceneflow.sh base
SCENEFLOW_DIR=$VDFNET_DATA bash gwcnet_vdf/train_gwcnet_sceneflow.sh render
# to keep a run alive after logout: nohup ... & disown

# 4) zero-shot cross-domain evaluation (eval sets under $WORKSPACE/eval_data/{ETH3D,KITTI,Middlebury})
cd $WORKSPACE/GwcNet_base   && python $VDFNET_ROOT/gwcnet_vdf/eval_gwcnet_generalization.py \
    --ckpt $VDFNET_CKPT/gwc_base/<best>.ckpt   --tag base
cd $WORKSPACE/GwcNet_render && python $VDFNET_ROOT/gwcnet_vdf/eval_gwcnet_generalization.py \
    --ckpt $VDFNET_CKPT/gwc_render/<best>.ckpt --tag render
```

## Notes / gotchas

- **Official GwcNet head shape**: the patch assumes the standard head
  `squeeze; F.softmax(cost, 1); disparity_regression(pred, maxdisp)`. If the
  upstream version differs, the patch **errors rather than failing silently** —
  follow the message and edit by hand (head lines only).
- **best checkpoint naming**: official GwcNet saves per-epoch
  `checkpoint_0000XX.ckpt` with no `best`. Pick the epoch with the lowest
  SceneFlow validation EPE (see `train_*.log`), or use the last epoch.
- **eval data layout**: `eval_gwcnet_generalization.py` follows the path
  assumptions of `eval_psmnet_generalization.py`
  (`eval_data/ETH3D/two_view_training/<scene>/im0.png`, etc.). If your eval_data
  layout differs, adjust `--eval_root` or the sub-paths in the script.
- **Fairness self-check**: before training, `diff` the `main.py` / dataloader /
  loss of the two repo copies to confirm that **only** the disparity head in
  `models/gwcnet.py` differs.

## Related

- Same-protocol PSMNet / AANet kits: `psmnet_vdf/`, `aanet_vdf/`.
