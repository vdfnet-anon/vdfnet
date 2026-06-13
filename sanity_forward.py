#!/usr/bin/env python3
"""
sanity_forward.py — quick 1-batch forward check for a patched backbone,
BEFORE committing to a multi-hour training run.

Verifies the disparityrender head produces sane disparities (finite, in
[0, maxdisp), non-degenerate). Catches the #1 failure mode: cost-volume
disparity-axis orientation disagreeing with disparityrender's internal flip,
which would silently train a wrong model for hours.

Run INSIDE the patched repo (so its `models`/`nets` import resolves).
$VDFNET_ROOT points at this repo; $WORKSPACE holds the cloned backbones:

  # GwcNet (run inside $WORKSPACE/GwcNet_render)
  CUDA_VISIBLE_DEVICES=0 python $VDFNET_ROOT/sanity_forward.py gwcnet

  # PSMNet (run inside $WORKSPACE/PSMNet_render)
  CUDA_VISIBLE_DEVICES=0 python $VDFNET_ROOT/sanity_forward.py psmnet

  # AANet (run inside $WORKSPACE/aanet_render)
  CUDA_VISIBLE_DEVICES=0 python $VDFNET_ROOT/sanity_forward.py aanet
"""
import sys
import torch


def build(backbone, maxdisp=192):
    if backbone == "gwcnet":
        from models.gwcnet import GwcNet
        return GwcNet(maxdisp, use_concat_volume=True)
    if backbone == "psmnet":
        from models.stackhourglass import PSMNet
        return PSMNet(maxdisp)
    if backbone == "aanet":
        # AANet's nets/ must take priority over the project root nets/ (stereorf).
        # Insert cwd (the patched backbone repo, e.g. $WORKSPACE/aanet_render)
        # at front of sys.path.
        import sys, os
        sys.path.insert(0, os.getcwd())
        from importlib import import_module
        AANet = import_module('nets.aanet').AANet
        return AANet(
            maxdisp, num_downsample=2, feature_type='aanet',
            no_feature_mdconv=False, feature_pyramid=False,
            feature_pyramid_network=True, feature_similarity='correlation',
            aggregation_type='adaptive', num_scales=3, num_fusions=6,
            num_stage_blocks=1, num_deform_blocks=3,
            no_intermediate_supervision=False, refinement_type='stereodrnet',
            mdconv_dilation=2, deformable_groups=2,
        )
    raise SystemExit(f"unknown backbone '{backbone}' (use gwcnet|psmnet|aanet)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    backbone = sys.argv[1].lower()
    maxdisp = int(sys.argv[2]) if len(sys.argv) > 2 else 192

    model = build(backbone, maxdisp).cuda().eval()

    params = dict(model.named_parameters())
    if 'density_temperature' in params:
        print(f"density_temperature = {params['density_temperature'].item():.4f}  "
              "(expected 1.0 before training)")
    else:
        print("[WARN] no density_temperature param found — is this the patched (render) repo?")

    # small even dims (PSMNet/GwcNet need /16, AANet /48); 384x768 is safe for all
    H, W = 384, 768
    L = torch.rand(1, 3, H, W).cuda()
    R = torch.rand(1, 3, H, W).cuda()
    with torch.no_grad():
        out = model(L, R)
    d = out[-1] if isinstance(out, (list, tuple)) else out
    if d.dim() == 4:
        d = d.squeeze(1)

    finite = bool(torch.isfinite(d).all())
    dmin, dmax, dmean = float(d.min()), float(d.max()), float(d.mean())
    std = float(d.std())
    print(f"output shape: {tuple(d.shape)}  (expect [1, {H}, {W}])")
    print(f"disp range: min={dmin:.3f}  max={dmax:.3f}  mean={dmean:.3f}  std={std:.3f}")

    ok = finite and dmin >= -1e-3 and dmax < maxdisp and std > 1e-4
    print("-" * 56)
    if ok:
        print(f"[PASS] {backbone}: disparities finite, in [0,{maxdisp}), non-degenerate.")
        print("       Orientation looks correct -> safe to start training.")
    else:
        print(f"[FAIL] {backbone}: SUSPICIOUS output.")
        reasons = []
        if not finite: reasons.append("non-finite (NaN/Inf)")
        if dmin < -1e-3: reasons.append(f"negative disp ({dmin:.3f})")
        if dmax >= maxdisp: reasons.append(f"disp >= maxdisp ({dmax:.3f})")
        if std <= 1e-4: reasons.append("degenerate/constant output")
        print("       reasons:", "; ".join(reasons))
        print("       -> likely cost-volume axis vs disparityrender.flip mismatch.")
        print("       Do NOT train yet; paste this output back.")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
