"""
Reproduce paper Table I (disparityrender ablation on SceneFlow test set, IGEV backbone).

Loads the three released IGEV checkpoints and evaluates each on the SceneFlow
test split, then prints a table in the same layout as the paper:

    Method                      EPE(px)   1-ER(%)   3-ER(%)
    soft-argmin (baseline)      0.4813    5.29      2.50
    +disparityrender            0.4790    5.38      2.51
    +density_temperature        0.4686    5.24      2.45

Usage (from the igev_baseline/ directory):
    SCENEFLOW_DIR=/path/to/SceneFlow python reproduce_table1.py \
        --softargmin <expA.pth> --render <expB.pth> --render_temp <ft4.pth>

If a checkpoint path is omitted, that row is skipped. valid_iters defaults to 32
(same as the paper's evaluation protocol).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

import argparse
import importlib
import torch
from evaluate_stereo import validate_sceneflow


def build_args(max_disp=192):
    """Minimal arg namespace to construct the full unmodified IGEVStereo network."""
    return argparse.Namespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4,
        n_downsample=2, n_gru_layers=3, max_disp=max_disp,
        mixed_precision=False, precision_dtype='float32',
        shared_backbone=False, slow_fast_gru=False, valid_iters=32,
    )


def load_model(ckpt_path, variant):
    """variant: 'soft-argmin' -> original IGEV; 'render'/'render_temp' -> render IGEV."""
    module_name = 'igev_stereo_original' if variant == 'soft-argmin' else 'igev_stereo'
    module = importlib.import_module(module_name)
    model = module.IGEVStereo(build_args())
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # render checkpoints without a learned temperature (Exp B) legitimately miss
    # density_temperature; everything else must match.
    benign = {'density_temperature'}
    hard_missing = [k for k in missing if k.split('.')[-1] not in benign]
    if hard_missing or unexpected:
        print(f"  [warn] unexpected={unexpected}, hard_missing={hard_missing}")
    return model.cuda().eval()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--softargmin', help='Exp A checkpoint (soft-argmin baseline)')
    p.add_argument('--render', help='Exp B checkpoint (+disparityrender)')
    p.add_argument('--render_temp', help='ft4 checkpoint (+density_temperature, flagship)')
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--mixed_precision', action='store_true')
    args = p.parse_args()

    if not os.environ.get('SCENEFLOW_DIR'):
        print("[error] SCENEFLOW_DIR is not set. Point it at your SceneFlow root, e.g.:")
        print("        SCENEFLOW_DIR=/path/to/SceneFlow python reproduce_table1.py ...")
        sys.exit(1)

    # (label, ckpt, variant, paper reference values)
    rows_spec = [
        ('soft-argmin (baseline)', args.softargmin, 'soft-argmin', (0.4813, 5.29, 2.50)),
        ('+disparityrender',       args.render,     'render',      (0.4790, 5.38, 2.51)),
        ('+density_temperature',   args.render_temp, 'render_temp', (0.4686, 5.24, 2.45)),
    ]

    results = []
    for label, ckpt, variant, paper in rows_spec:
        if not ckpt:
            continue
        print(f"\n=== Evaluating: {label}  ({ckpt}) ===")
        model = load_model(ckpt, variant)
        m = validate_sceneflow(model, iters=args.iters, mixed_prec=args.mixed_precision)
        results.append((label, m['scene-disp-epe'], m['scene-disp-1er'], m['scene-disp-d1'], paper))
        del model
        torch.cuda.empty_cache()

    # ---- print paper-style table (measured vs paper) ----
    print("\n" + "=" * 72)
    print("Table I — disparityrender ablation on SceneFlow test set (IGEV backbone)")
    print("=" * 72)
    print(f"{'Method':<26}{'EPE(px)':>10}{'1-ER(%)':>10}{'3-ER(%)':>10}   {'[paper]':>20}")
    print("-" * 72)
    for label, epe, er1, d1, paper in results:
        pe, p1, p3 = paper
        print(f"{label:<26}{epe:>10.4f}{er1:>10.2f}{d1:>10.2f}   "
              f"(paper {pe:.4f}/{p1:.2f}/{p3:.2f})")
    print("=" * 72)
    print("Measured values should match the paper column within run-to-run variance.")


if __name__ == '__main__':
    main()

