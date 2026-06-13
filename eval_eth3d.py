"""
Evaluate VDFNet predictions on ETH3D two-view stereo training set.
Metrics: EPE, bad-1.0, bad-2.0, bad-4.0, RMS

Usage:
    python eval_eth3d.py --data_dir /path/to/ETH3D --pred_dir /path/to/ETH3D/pred
"""
import argparse
import os
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--pred_dir', default=None)
args = parser.parse_args()

if args.pred_dir is None:
    args.pred_dir = os.path.join(args.data_dir, 'pred')


def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().rstrip().decode('ascii')
        color = (header == 'PF')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('ascii'))
        width, height = map(int, dim_match.groups())
        scale = float(f.readline().decode('ascii').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data


def evaluate_scene(gt_path, pred_path, mask_path=None):
    gt = read_pfm(gt_path).astype(np.float32)
    pred = read_pfm(pred_path).astype(np.float32)

    # Valid mask: GT > 0 and not inf
    valid = (gt > 0) & np.isfinite(gt)

    # Non-occluded mask
    if mask_path and os.path.exists(mask_path):
        import cv2
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            valid = valid & (mask > 0)

    if valid.sum() == 0:
        return None

    err = np.abs(gt[valid] - pred[valid])
    epe = err.mean()
    bad1 = (err > 1.0).mean() * 100
    bad2 = (err > 2.0).mean() * 100
    bad4 = (err > 4.0).mean() * 100
    rms = np.sqrt((err ** 2).mean())
    return epe, bad1, bad2, bad4, rms


train_dir = os.path.join(args.data_dir, 'two_view_training')
scenes = sorted(os.listdir(train_dir))

results = []
print(f'{"Scene":<30} {"EPE":>6} {"bad1":>6} {"bad2":>6} {"bad4":>6} {"RMS":>6}')
print('-' * 62)

for scene in scenes:
    gt_path = os.path.join(train_dir, scene, 'disp0GT.pfm')
    pred_path = os.path.join(args.pred_dir, 'two_view_training', scene, 'im0.pfm')
    mask_path = os.path.join(train_dir, scene, 'mask0nocc.png')

    if not os.path.exists(pred_path):
        print(f'{scene:<30} {"MISSING":>6}')
        continue

    r = evaluate_scene(gt_path, pred_path, mask_path)
    if r is None:
        continue
    epe, bad1, bad2, bad4, rms = r
    results.append(r)
    print(f'{scene:<30} {epe:6.3f} {bad1:6.2f} {bad2:6.2f} {bad4:6.2f} {rms:6.3f}')

if results:
    results = np.array(results)
    mean = results.mean(axis=0)
    print('-' * 62)
    print(f'{"Mean":<30} {mean[0]:6.3f} {mean[1]:6.2f} {mean[2]:6.2f} {mean[3]:6.2f} {mean[4]:6.3f}')
