"""
Evaluate VDFNet predictions on Middlebury v3 training set.
Metrics: EPE, bad-1.0, bad-2.0, bad-4.0, RMS

Usage:
    python eval_middlebury.py --data_dir /path/to/Middlebury --pred_dir /path/to/results/Middlebury --resolution H
"""
import argparse
import os
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--pred_dir', required=True)
parser.add_argument('--resolution', default='H', choices=['H', 'F', 'Q'])
args = parser.parse_args()


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

    valid = (gt > 0) & np.isfinite(gt)

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


train_dir = os.path.join(args.data_dir, 'MiddEval3', f'training{args.resolution}')
scenes = sorted(os.listdir(train_dir))

results = []
print(f'{"Scene":<20} {"EPE":>8} {"bad1":>6} {"bad2":>6} {"bad4":>6} {"RMS":>8}')
print('-' * 56)

for scene in scenes:
    gt_path = os.path.join(train_dir, scene, 'disp0GT.pfm')
    pred_path = os.path.join(args.pred_dir, 'MiddEval3', f'training{args.resolution}', scene, 'im0.pfm')
    mask_path = os.path.join(train_dir, scene, 'mask0nocc.png')

    if not os.path.exists(gt_path):
        continue
    if not os.path.exists(pred_path):
        print(f'{scene:<20} {"MISSING":>8}')
        continue

    r = evaluate_scene(gt_path, pred_path, mask_path)
    if r is None:
        continue
    epe, bad1, bad2, bad4, rms = r
    results.append(r)
    print(f'{scene:<20} {epe:8.3f} {bad1:6.2f} {bad2:6.2f} {bad4:6.2f} {rms:8.3f}')

if results:
    results = np.array(results)
    mean = results.mean(axis=0)
    print('-' * 56)
    print(f'{"Mean":<20} {mean[0]:8.3f} {mean[1]:6.2f} {mean[2]:6.2f} {mean[3]:6.2f} {mean[4]:8.3f}')
