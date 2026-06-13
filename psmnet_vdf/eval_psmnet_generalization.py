#!/usr/bin/env python3
"""
eval_psmnet_generalization.py — zero-shot generalization for the PSMNet
second-backbone ablation, using the SAME metric definitions as the paper's
existing tables (copied from scripts/eval_psmnet_generalization.py), so the
numbers go side-by-side with Tables VII/VIII/IX.

Evaluates an official PSMNet checkpoint (soft-argmin OR disparityrender — the
model class is the same after patching) on ETH3D / KITTI 2015 / Middlebury H,
SceneFlow-pretrained, no fine-tuning.

Preprocessing mirrors PSMNet's own submission.py (ImageNet normalize, pad to
a multiple of 16 with top/right padding, model(L,R) -> single disp at eval).

Usage (run inside the PSMNet repo so `models` imports resolve):
    cd $WORKSPACE/PSMNet_base   && python $VDFNET_ROOT/psmnet_vdf/eval_psmnet_generalization.py \
        --ckpt $VDFNET_CKPT/psm_base/checkpoint_9.tar   --tag base
    cd $WORKSPACE/PSMNet_render && python $VDFNET_ROOT/psmnet_vdf/eval_psmnet_generalization.py \
        --ckpt $VDFNET_CKPT/psm_render/checkpoint_9.tar --tag render

Datasets expected at $VDFNET_EVAL_DATA/{ETH3D,KITTI,Middlebury}
(same layout the paper's eval scripts use).
"""
import sys
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
from tqdm import tqdm


# ---------------------------------------------------------------------------
# loaders / metrics — IDENTICAL to scripts/eval_psmnet_generalization.py
# ---------------------------------------------------------------------------
def load_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            channels = 3
        elif header == 'Pf':
            channels = 1
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        width, height = int(dim_match.group(1)), int(dim_match.group(2))
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, channels) if channels > 1 else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data).copy()
    return data


# PSMNet eval transform: ImageNet normalize (matches submission.py).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def load_image(path):
    img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - _MEAN[i]) / _STD[i]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()


def compute_metrics(pred, gt, mask):
    epe = np.abs(pred - gt)
    epe_valid = epe[mask]
    results = {'EPE': np.mean(epe_valid)}
    for t in [0.5, 1.0, 2.0, 3.0, 4.0]:
        results[f'bad-{t}'] = (epe_valid > t).mean() * 100
    d1 = ((epe_valid > 3) & (epe_valid / np.maximum(np.abs(gt[mask]), 1e-6) > 0.05)).mean() * 100
    results['D1-all'] = d1
    return results


@torch.no_grad()
def run_model(model, im0_path, im1_path):
    imgL = load_image(im0_path)
    imgR = load_image(im1_path)
    H, W = imgL.shape[2], imgL.shape[3]
    # PSMNet pads to a multiple of 16 with TOP/RIGHT padding (submission.py).
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    imgL = F.pad(imgL, (0, pad_w, pad_h, 0))
    imgR = F.pad(imgR, (0, pad_w, pad_h, 0))
    out = model(imgL, imgR)            # eval -> single disp [B,H,W]
    if isinstance(out, (list, tuple)):
        out = out[-1]
    pred = torch.squeeze(out).cpu().numpy()
    # remove the top/right padding
    if pad_h or pad_w:
        pred = pred[pad_h:, :pred.shape[1] - pad_w if pad_w else pred.shape[1]]
    return pred[:H, :W]


def _avg(metrics):
    if not metrics:
        print("  WARNING: no valid samples!")
        return {}
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}


def eval_eth3d(model, root):
    img_dir = os.path.join(root, 'two_view_training')
    scenes = sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)))
    ms = []
    for s in tqdm(scenes, desc='ETH3D'):
        im0, im1 = os.path.join(img_dir, s, 'im0.png'), os.path.join(img_dir, s, 'im1.png')
        gtp = os.path.join(root, s, 'disp0GT.pfm')
        mkp = os.path.join(root, s, 'mask0nocc.png')
        if not all(os.path.exists(p) for p in [im0, im1, gtp, mkp]):
            continue
        pred = run_model(model, im0, im1)
        gt = load_pfm(gtp)
        mask = np.array(Image.open(mkp)) > 0
        if pred.shape != gt.shape:
            pred = pred[:gt.shape[0], :gt.shape[1]]
        ms.append(compute_metrics(pred, gt, mask & np.isfinite(gt) & (gt > 0)))
    return _avg(ms)


def eval_kitti(model, root):
    base = os.path.join(root, 'KITTI_2015', 'training')
    img2 = sorted(glob(os.path.join(base, 'image_2', '*.png')))
    disp_dir = os.path.join(base, 'disp_noc_0')
    ms = []
    for l in tqdm(img2, desc='KITTI'):
        r = l.replace('image_2', 'image_3')
        gtp = os.path.join(disp_dir, os.path.basename(l))
        if not os.path.exists(gtp):
            continue
        pred = run_model(model, l, r)
        gt = np.array(Image.open(gtp)).astype(np.float32) / 256.0
        if pred.shape != gt.shape:
            pred = pred[:gt.shape[0], :gt.shape[1]]
        ms.append(compute_metrics(pred, gt, gt > 0))
    return _avg(ms)


def eval_middlebury(model, root):
    base = os.path.join(root, 'trainingH')
    scenes = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    ms = []
    for s in tqdm(scenes, desc='Middlebury'):
        im0, im1 = os.path.join(base, s, 'im0.png'), os.path.join(base, s, 'im1.png')
        gtp = os.path.join(base, s, 'disp0GT.pfm')
        mkp = os.path.join(base, s, 'mask0nocc.png')
        if not all(os.path.exists(p) for p in [im0, im1, gtp]):
            continue
        pred = run_model(model, im0, im1)
        gt = load_pfm(gtp)
        mask = (np.array(Image.open(mkp)) > 0) if os.path.exists(mkp) else (gt > 0)
        if pred.shape != gt.shape:
            pred = pred[:gt.shape[0], :gt.shape[1]]
        ms.append(compute_metrics(pred, gt, mask & (gt > 0) & (gt < 1e4)))
    return _avg(ms)


def build_psmnet(maxdisp):
    """Import the (possibly patched) official PSMNet from the cwd repo."""
    import importlib
    sh = importlib.import_module('models.stackhourglass')
    return sh.PSMNet(maxdisp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='model', help='label for this run, e.g. base / render')
    ap.add_argument('--maxdisp', type=int, default=192)
    ap.add_argument('--eval_root', default=os.environ.get('VDFNET_EVAL_DATA', 'data/eval_data'))
    args = ap.parse_args()

    model = build_psmnet(args.maxdisp)
    state = torch.load(args.ckpt, map_location='cpu')
    sd = state.get('state_dict', state.get('model', state))
    sd = {k.replace('module.', ''): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{args.tag}] loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    if any('density_temperature' in m for m in missing):
        print(f"[{args.tag}] NOTE: density_temperature missing -> this is a soft-argmin (base) ckpt")
    model = model.cuda().eval()

    print("=" * 64)
    print(f"PSMNet [{args.tag}] zero-shot generalization (SceneFlow-pretrained)")
    print("=" * 64)
    eth = eval_eth3d(model, os.path.join(args.eval_root, 'ETH3D'))
    kit = eval_kitti(model, os.path.join(args.eval_root, 'KITTI'))
    mid = eval_middlebury(model, os.path.join(args.eval_root, 'Middlebury'))

    def line(name, d, keys):
        cells = '  '.join(f"{k}={d.get(k, float('nan')):.4f}" for k in keys)
        print(f"  [{name}] {cells}")

    print(f"\n----- RESULT [{args.tag}] -----")
    line('ETH3D',      eth, ['EPE', 'bad-0.5', 'bad-1.0', 'bad-2.0'])
    line('KITTI2015',  kit, ['EPE', 'D1-all', 'bad-2.0', 'bad-4.0'])
    line('MiddleburyH', mid, ['EPE', 'bad-2.0', 'bad-4.0'])
    print("\nRun this for BOTH --tag base and --tag render, then compare:")
    print("  success = in-domain EPE within ~0.02 AND render better on >=1 of "
          "{ETH3D EPE/bad, KITTI EPE/D1-all}.")


if __name__ == '__main__':
    main()
