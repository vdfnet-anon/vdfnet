"""
Zero-shot generalization evaluation: all v6 models on ETH3D, Middlebury, and KITTI.
Models: Exp A (soft-argmin), Exp B (disparityrender), ft2, ft3, ft4
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import torch
import torch.nn as nn
import importlib
from evaluate_stereo import validate_eth3d, validate_middlebury, validate_kitti


MODELS = [
    ('Exp A (soft-argmin)', 'igev_stereo_original', 'checkpoints/200000_igev_original.pth'),
    ('Exp B (render 200k)', 'igev_stereo', 'checkpoints/v6_igev_render/200000_igev_render.pth'),
    ('ft2 (lr=5e-5, 50k)', 'igev_stereo', 'checkpoints/v6_igev_render_ft2/igev_render_ft2.pth'),
    ('ft3 (lr=3e-5, 50k)', 'igev_stereo', 'checkpoints/v6_igev_render_ft3/igev_render_ft3.pth'),
    ('ft4 (lr=2e-5, 100k)', 'igev_stereo', 'checkpoints/v6_igev_render_ft4/igev_render_ft4.pth'),
]


def load_model(module_name, ckpt_path, args):
    module = importlib.import_module(f'core.{module_name}')
    model = module.IGEVStereo(args)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=32)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--models', nargs='+', type=int, default=None,
                        help='Indices of models to evaluate (0-4). Default: all')

    # Model args
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--precision_dtype', default='float32')
    args = parser.parse_args()

    model_indices = args.models if args.models else list(range(len(MODELS)))

    print("=" * 70)
    print("Zero-shot Generalization Evaluation (SceneFlow -> ETH3D / Middlebury / KITTI)")
    print("=" * 70)

    all_results = {}

    for idx in model_indices:
        name, module_name, ckpt_path = MODELS[idx]
        print(f"\n[{idx}] {name}")
        print(f"    Checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"    SKIP: checkpoint not found")
            continue

        model = load_model(module_name, ckpt_path, args)

        print("    Evaluating ETH3D...")
        r_eth3d = validate_eth3d(model, iters=args.iters, mixed_prec=args.mixed_precision)
        print("    Evaluating Middlebury...")
        r_mid = validate_middlebury(model, iters=args.iters, split='H', mixed_prec=args.mixed_precision)
        print("    Evaluating KITTI...")
        r_kitti = validate_kitti(model, iters=args.iters, mixed_prec=args.mixed_precision)

        all_results[name] = {'eth3d': r_eth3d, 'middlebury': r_mid, 'kitti': r_kitti}
        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    header = (f"{'Model':<25} {'ETH3D EPE':<11} {'bad-0.5':<9} {'bad-1.0':<9} {'bad-2.0':<9} "
              f"{'Mid EPE':<10} {'bad-2.0':<9} {'bad-4.0':<9} "
              f"{'KITTI EPE':<11} {'D1-all':<9} {'bad-2':<9} {'bad-4':<9}")
    print(header)
    print("-" * 120)

    for name, results in all_results.items():
        eth_epe   = results['eth3d'].get('eth3d-epe',    float('nan'))
        eth_b05   = results['eth3d'].get('eth3d-bad05',  float('nan'))
        eth_b10   = results['eth3d'].get('eth3d-bad10',  float('nan'))
        eth_b20   = results['eth3d'].get('eth3d-bad20',  float('nan'))
        mid_epe   = results['middlebury'].get('middleburyH-epe',   float('nan'))
        mid_b20   = results['middlebury'].get('middleburyH-bad20', float('nan'))
        mid_b40   = results['middlebury'].get('middleburyH-bad40', float('nan'))
        kitti_epe = results['kitti'].get('kitti-epe',  float('nan'))
        kitti_d1  = results['kitti'].get('kitti-d1',   float('nan'))
        kitti_b2  = results['kitti'].get('kitti-bad2', float('nan'))
        kitti_b4  = results['kitti'].get('kitti-bad4', float('nan'))
        print(f"{name:<25} {eth_epe:<11.4f} {eth_b05:<9.2f} {eth_b10:<9.2f} {eth_b20:<9.2f} "
              f"{mid_epe:<10.4f} {mid_b20:<9.2f} {mid_b40:<9.2f} "
              f"{kitti_epe:<11.4f} {kitti_d1:<9.2f} {kitti_b2:<9.2f} {kitti_b4:<9.2f}")

    # Delta vs Exp A
    if 'Exp A (soft-argmin)' in all_results:
        base = all_results['Exp A (soft-argmin)']
        print(f"\n{'--- Delta vs Exp A ---':<25} {'ΔETH3D EPE':<11} {'Δbad-1.0':<9} {'ΔMid EPE':<10} {'Δbad-2.0':<9} {'ΔKITTI EPE':<11} {'ΔD1-all':<9}")
        print("-" * 80)
        for name, results in all_results.items():
            if name == 'Exp A (soft-argmin)':
                continue
            d_eth_epe   = results['eth3d'].get('eth3d-epe', 0)   - base['eth3d'].get('eth3d-epe', 0)
            d_eth_b10   = results['eth3d'].get('eth3d-bad10', 0)  - base['eth3d'].get('eth3d-bad10', 0)
            d_mid_epe   = results['middlebury'].get('middleburyH-epe', 0)   - base['middlebury'].get('middleburyH-epe', 0)
            d_mid_b20   = results['middlebury'].get('middleburyH-bad20', 0) - base['middlebury'].get('middleburyH-bad20', 0)
            d_kitti_epe = results['kitti'].get('kitti-epe', 0) - base['kitti'].get('kitti-epe', 0)
            d_kitti_d1  = results['kitti'].get('kitti-d1', 0)  - base['kitti'].get('kitti-d1', 0)
            print(f"{name:<25} {d_eth_epe:<+11.4f} {d_eth_b10:<+9.2f} {d_mid_epe:<+10.4f} {d_mid_b20:<+9.2f} {d_kitti_epe:<+11.4f} {d_kitti_d1:<+9.2f}")
        print("\nNegative = better than Exp A")

    print("=" * 70)


if __name__ == '__main__':
    main()
