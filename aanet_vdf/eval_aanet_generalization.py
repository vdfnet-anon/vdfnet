#!/usr/bin/env python3
"""
eval_aanet_generalization.py — zero-shot generalization for the AANet
second-backbone ablation, using the SAME metric definitions as the paper's
existing tables, so the numbers go side-by-side with Tables VII/VIII/IX.

Evaluates an official AANet checkpoint (soft-argmin OR disparityrender — the
model class is the same after patching nets/estimation.py) on
ETH3D / KITTI 2015 / Middlebury H, SceneFlow-pretrained, no fine-tuning.

Preprocessing mirrors AANet's own predict.py (ImageNet normalize, pad to the
configured size, aanet(L,R)[-1] -> final-scale disp [B,H,W]). Model is built
with the SAME args as train_aanet_sceneflow.sh (feature_type aanet + FPN,
num_scales 3, num_downsample 2) so the checkpoint loads cleanly.

Usage (run inside the AANet repo so `nets` imports resolve):
    cd $WORKSPACE/aanet_base   && python $VDFNET_ROOT/aanet_vdf/eval_aanet_generalization.py \
        --ckpt $VDFNET_CKPT/aanet_base/aanet_best.pth   --tag base
    cd $WORKSPACE/aanet_render && python $VDFNET_ROOT/aanet_vdf/eval_aanet_generalization.py \
        --ckpt $VDFNET_CKPT/aanet_render/aanet_best.pth --tag render

Datasets expected at $VDFNET_EVAL_DATA/{ETH3D,KITTI,Middlebury}.
"""
import sys
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import re
import numpy as np
import torch
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
    # AANet downsamples by up to 12x; pad to a multiple of 48 to be safe
    # (predict.py pads to a fixed img_height/width; multiple-of-48 is the
    # general-purpose equivalent for arbitrary eval resolutions).
    factor = 48
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    imgL = F.pad(imgL, (0, pad_w, pad_h, 0))
    imgR = F.pad(imgR, (0, pad_w, pad_h, 0))
    out = model(imgL, imgR)            # list of pyramid disps
    if isinstance(out, (list, tuple)):
        out = out[-1]                  # final full-res disparity [B,H,W]
    # upsample if the final scale is below input resolution (predict.py logic)
    if out.dim() == 3:
        out = out.unsqueeze(1)
    if out.size(-1) < imgL.size(-1):
        scale = imgL.size(-1) / out.size(-1)
        out = F.interpolate(out, (imgL.size(-2), imgL.size(-1)),
                            mode='bilinear', align_corners=False) * scale
    pred = out.squeeze().cpu().numpy()
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


def build_aanet(maxdisp):
    """Build official AANet with the SAME args as train_aanet_sceneflow.sh."""
    import importlib
    nets = importlib.import_module('nets')
    return nets.AANet(
        maxdisp,
        num_downsample=2,
        feature_type='aanet',
        no_feature_mdconv=False,
        feature_pyramid=False,
        feature_pyramid_network=True,
        feature_similarity='correlation',
        aggregation_type='adaptive',
        num_scales=3,
        num_fusions=6,
        num_stage_blocks=1,
        num_deform_blocks=3,
        no_intermediate_supervision=False,
        refinement_type='stereodrnet',
        mdconv_dilation=2,
        deformable_groups=2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='model', help='label for this run, e.g. base / render')
    ap.add_argument('--maxdisp', type=int, default=192)
    ap.add_argument('--eval_root', default=os.environ.get('VDFNET_EVAL_DATA', 'data/eval_data'))
    args = ap.parse_args()

    model = build_aanet(args.maxdisp)
    state = torch.load(args.ckpt, map_location='cpu')
    sd = state.get('state_dict', state.get('model', state))
    sd = {k.replace('module.', ''): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{args.tag}] loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    if any('density_temperature' in m for m in missing):
        print(f"[{args.tag}] NOTE: density_temperature missing -> this is a soft-argmin (base) ckpt")
    model = model.cuda().eval()

    print("=" * 64)
    print(f"AANet [{args.tag}] zero-shot generalization (SceneFlow-pretrained)")
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
