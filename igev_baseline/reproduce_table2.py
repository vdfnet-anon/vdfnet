"""
Reproduce paper Table II (IGEV rows) -- zero-shot cross-domain generalization.

Loads the released IGEV checkpoints and evaluates each on the zero-shot
cross-domain benchmarks (ETH3D / Middlebury H; KITTI optional if present),
then prints a table comparing soft-argmin (base) vs disparityrender (render)
against the paper's reported numbers.

Paper Table II, IGEV rows:
    Head          ETH3D EPE    KITTI D1    Mid. EPE
    soft-argmin   0.322        6.67        0.848
    render        0.279        5.96        0.885

Usage (from the igev_baseline/ directory, with /data/ETH3D, /data/Middlebury,
and optionally /data/KITTI in place):
    python reproduce_table2.py \
        --softargmin <expA.pth> --render <expB.pth> [--render_temp <ft4.pth>] \
        [--datasets eth3d middlebury_H kitti]

Any checkpoint flag omitted -> that row is skipped. By default KITTI is included
only if --datasets lists it (it requires the registered-login download).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

import argparse
import importlib
import torch
from evaluate_stereo import (validate_eth3d, validate_kitti,
                             validate_middlebury, validate_sceneflow)


def build_args(max_disp=192):
    return argparse.Namespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4,
        n_downsample=2, n_gru_layers=3, max_disp=max_disp,
        mixed_precision=False, precision_dtype='float32',
        shared_backbone=False, slow_fast_gru=False, valid_iters=32,
    )


def load_model(ckpt_path, variant):
    module_name = 'igev_stereo_original' if variant == 'soft-argmin' else 'igev_stereo'
    module = importlib.import_module(module_name)
    model = module.IGEVStereo(build_args())
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    benign = {'density_temperature'}
    hard_missing = [k for k in missing if k.split('.')[-1] not in benign]
    if hard_missing or unexpected:
        print(f"  [warn] unexpected={unexpected}, hard_missing={hard_missing}")
    return model.cuda().eval()

def eval_one(model, datasets, iters, mixed):
    """Return dict of headline metrics for the requested datasets."""
    out = {}
    if 'eth3d' in datasets:
        m = validate_eth3d(model, iters=iters, mixed_prec=mixed)
        out['eth3d-epe'] = m['eth3d-epe']
    if 'kitti' in datasets:
        m = validate_kitti(model, iters=iters, mixed_prec=mixed)
        out['kitti-d1'] = m['kitti-d1']
    for split in 'FHQ':
        key = f'middlebury_{split}'
        if key in datasets:
            m = validate_middlebury(model, iters=iters, split=split, mixed_prec=mixed)
            out[f'mid{split}-epe'] = m[f'middlebury{split}-epe']
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--softargmin', help='Exp A checkpoint (soft-argmin baseline)')
    p.add_argument('--render', help='Exp B checkpoint (+disparityrender, 0.4790)')
    p.add_argument('--render_temp', help='ft4 checkpoint (+density_temperature, flagship)')
    p.add_argument('--datasets', nargs='+', default=['eth3d', 'middlebury_H'],
                   help='subset of: eth3d kitti middlebury_F middlebury_H middlebury_Q')
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--mixed_precision', action='store_true')
    args = p.parse_args()

    variants = [
        ('soft-argmin (base)', args.softargmin, 'soft-argmin'),
        ('+disparityrender',   args.render,     'render'),
        ('+density_temperature', args.render_temp, 'render_temp'),
    ]
    # paper Table II IGEV rows (base, render) for the headline metrics
    paper = {
        'soft-argmin (base)': {'eth3d-epe': 0.322, 'kitti-d1': 6.67, 'midH-epe': 0.848},
        '+disparityrender':   {'eth3d-epe': 0.279, 'kitti-d1': 5.96, 'midH-epe': 0.885},
    }

    results = []
    for label, ckpt, variant in variants:
        if not ckpt:
            continue
        print(f"\n=== Evaluating: {label}  ({ckpt}) ===")
        model = load_model(ckpt, variant)
        metrics = eval_one(model, args.datasets, args.iters, args.mixed_precision)
        results.append((label, metrics))
        print(f"  -> {metrics}")
        del model
        torch.cuda.empty_cache()

    # ---- print paper-style cross-domain table ----
    cols = []
    if 'eth3d' in args.datasets:       cols.append(('eth3d-epe', 'ETH3D EPE'))
    if 'kitti' in args.datasets:       cols.append(('kitti-d1',  'KITTI D1'))
    if 'middlebury_H' in args.datasets: cols.append(('midH-epe', 'Mid.H EPE'))

    print("\n" + "=" * 78)
    print("Table II (IGEV rows) -- zero-shot cross-domain generalization")
    print("=" * 78)
    header = f"{'Head':<24}" + "".join(f"{name:>14}" for _, name in cols)
    print(header)
    print("-" * 78)
    for label, metrics in results:
        row = f"{label:<24}"
        for key, _ in cols:
            v = metrics.get(key)
            pv = paper.get(label, {}).get(key)
            cell = f"{v:.4f}" if v is not None else "-"
            if pv is not None:
                cell += f"({pv})"
            row += f"{cell:>14}"
        print(row)
    print("=" * 78)
    print("Paper values in parentheses. Measured should match within variance.")


if __name__ == '__main__':
    main()

