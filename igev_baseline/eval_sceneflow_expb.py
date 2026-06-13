"""Evaluate Exp B (disparityrender 200k, no density_temperature) on SceneFlow test set."""
import os
import sys
sys.path.insert(0, 'core')

import argparse
import torch
import importlib
from evaluate_stereo import validate_sceneflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/v6_igev_render/200000_igev_render.pth')
    parser.add_argument('--iters', type=int, default=32)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--precision_dtype', default='float32')
    parser.add_argument('--mixed_precision', action='store_true')
    args = parser.parse_args()

    module = importlib.import_module('igev_stereo')
    model = module.IGEVStereo(args)
    sd = torch.load(args.ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys (expected): {missing}")

    model.cuda().eval()
    print(f"Evaluating Exp B (disparityrender 200k): {args.ckpt}")
    validate_sceneflow(model, iters=args.iters, mixed_prec=args.mixed_precision)


if __name__ == '__main__':
    main()
