#!/usr/bin/env python3
"""
diagnose_render.py — pinpoint WHY disparityrender hurts KITTI D1 / Middlebury
while winning ETH3D, by bucketing errors by GT disparity and comparing
base (soft-argmin) vs render side by side on the SAME images.

Reports, per GT-disparity bucket:
  - signed error mean (pred - gt)   -> systematic BIAS (fixable) vs ~0 (variance)
  - abs error mean (EPE)
  - >3px error rate (D1 contribution)
  - sub-pixel <1px rate

Run TWICE (once per repo/ckpt) but it loads BOTH base and render in one go
if you pass both ckpts. Pure inference, no training touched.

Usage (from $WORKSPACE/GwcNet_render, which can import models for the patched
class; both ckpts load into the same patched class — base just lacks the
density_temperature param which load_state_dict(strict=False) handles, BUT
the base ckpt was trained with soft-argmin so we must load it into the
UNPATCHED class. Therefore: run base from $WORKSPACE/GwcNet_base, render from
$WORKSPACE/GwcNet_render, each with --tag, and this script appends to a shared
npz so a final --compare step prints the table.)

Simplest: run per-model, dump per-pixel (gt, pred) arrays, then compare.
    cd $WORKSPACE/GwcNet_base   && python diagnose_render.py --ckpt <base>   --tag base   --dump $VDFNET_DATA/diag_base.npz
    cd $WORKSPACE/GwcNet_render && python diagnose_render.py --ckpt <render> --tag render --dump $VDFNET_DATA/diag_render.npz
    python diagnose_render.py --compare $VDFNET_DATA/diag_base.npz $VDFNET_DATA/diag_render.npz
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from glob import glob
from tqdm import tqdm

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
BUCKETS = [(0, 16), (16, 32), (32, 48), (48, 64), (64, 96), (96, 192)]


def load_image(path):
    img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - _MEAN[i]) / _STD[i]
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].cuda()


def build_gwcnet(maxdisp):
    import importlib
    gm = importlib.import_module('models')
    if hasattr(gm, '__models__'):
        return gm.__models__['gwcnet-gc'](maxdisp)
    return importlib.import_module('models.gwcnet').GwcNet(maxdisp, use_concat_volume=True)


@torch.no_grad()
def run_model(model, im0, im1):
    imgL, imgR = load_image(im0), load_image(im1)
    H, W = imgL.shape[2], imgL.shape[3]
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    imgL = F.pad(imgL, [0, pw, 0, ph]); imgR = F.pad(imgR, [0, pw, 0, ph])
    out = model(imgL, imgR)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.squeeze().cpu().numpy()[:H, :W]


def collect_kitti(model, root):
    base = os.path.join(root, 'KITTI_2015', 'training')
    img2 = sorted(glob(os.path.join(base, 'image_2', '*.png')))
    disp_dir = os.path.join(base, 'disp_noc_0')
    gts, preds = [], []
    for l in tqdm(img2, desc='KITTI collect'):
        r = l.replace('image_2', 'image_3')
        gtp = os.path.join(disp_dir, os.path.basename(l))
        if not os.path.exists(gtp):
            continue
        pred = run_model(model, l, r)
        gt = np.array(Image.open(gtp)).astype(np.float32) / 256.0
        if pred.shape != gt.shape:
            pred = pred[:gt.shape[0], :gt.shape[1]]
        m = gt > 0
        gts.append(gt[m]); preds.append(pred[m])
    return np.concatenate(gts), np.concatenate(preds)


def compare(npz_base, npz_render):
    b = np.load(npz_base); r = np.load(npz_render)
    gt = b['gt']; pb = b['pred']; pr = r['pred']
    assert gt.shape == pb.shape == pr.shape, "pixel arrays misaligned"
    print(f"\n{'bucket':>10} | {'N':>8} | {'signed(base)':>12} {'signed(rndr)':>12} | "
          f"{'EPE(base)':>9} {'EPE(rndr)':>9} | {'>3px(base)':>10} {'>3px(rndr)':>10}")
    print("-" * 100)
    for lo, hi in BUCKETS:
        m = (gt >= lo) & (gt < hi)
        if m.sum() == 0:
            continue
        sb = (pb[m] - gt[m]); sr = (pr[m] - gt[m])
        eb = np.abs(sb); er = np.abs(sr)
        d3b = (eb > 3).mean() * 100; d3r = (er > 3).mean() * 100
        print(f"{lo:>4}-{hi:<4} | {m.sum():>8} | {sb.mean():>12.3f} {sr.mean():>12.3f} | "
              f"{eb.mean():>9.3f} {er.mean():>9.3f} | {d3b:>9.2f}% {d3r:>9.2f}%")
    # overall
    eb = np.abs(pb - gt); er = np.abs(pr - gt)
    print("-" * 100)
    print(f"{'ALL':>10} | {gt.size:>8} | {(pb-gt).mean():>12.3f} {(pr-gt).mean():>12.3f} | "
          f"{eb.mean():>9.3f} {er.mean():>9.3f} | {(eb>3).mean()*100:>9.2f}% {(er>3).mean()*100:>9.2f}%")
    print("\nINTERPRETATION:")
    print("  signed!=0 consistently  -> systematic BIAS (potentially fixable by recalibration)")
    print("  signed~0 but EPE/>3px up -> precision/variance loss (method tradeoff, not a bug)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt'); ap.add_argument('--tag', default='model')
    ap.add_argument('--maxdisp', type=int, default=192)
    ap.add_argument('--eval_root', default=os.environ.get('VDFNET_EVAL_DATA', 'data/eval_data'))
    ap.add_argument('--dump'); ap.add_argument('--compare', nargs=2)
    ap.add_argument('--show-temp', help='print density_temperature in a ckpt and exit')
    args = ap.parse_args()

    if args.show_temp:
        sd = torch.load(args.show_temp, map_location='cpu')
        sd = sd.get('model', sd.get('state_dict', sd))
        found = False
        for k, v in sd.items():
            if 'temperature' in k:
                print(f"{k} = {float(v.reshape(-1)[0]):.4f}")
                found = True
        if not found:
            print("no density_temperature found in ckpt")
        return

    if args.compare:
        compare(*args.compare); return

    model = build_gwcnet(args.maxdisp)
    state = torch.load(args.ckpt, map_location='cpu')
    sd = state.get('model', state.get('state_dict', state))
    sd = {k.replace('module.', ''): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    gt, pred = collect_kitti(model, os.path.join(args.eval_root, 'KITTI'))
    if args.dump:
        np.savez(args.dump, gt=gt, pred=pred)
        print(f"[{args.tag}] dumped {gt.size} pixels -> {args.dump}")


if __name__ == '__main__':
    main()
