"""
Inference speed & model complexity benchmark.
Measures: parameter count, FLOPs (via dummy forward), and FPS on standard resolution.
Models: Exp A (soft-argmin) vs Exp B (disparityrender)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('core')
import time
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn


MODELS = [
    ('Exp A (soft-argmin)', 'igev_stereo_original', 'checkpoints/200000_igev_original.pth'),
    ('Exp B (disparityrender)', 'igev_stereo', 'checkpoints/v6_igev_render/200000_igev_render.pth'),
]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model(module_name, ckpt_path, args):
    module = importlib.import_module(f'core.{module_name}')
    model = module.IGEVStereo(args)
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()
    return model


def benchmark_fps(model, H=480, W=640, iters=32, warmup=10, repeats=50):
    """Measure FPS with dummy input at given resolution."""
    dummy_left = torch.randn(1, 3, H, W).cuda()
    dummy_right = torch.randn(1, 3, H, W).cuda()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_left, dummy_right, iters=iters, test_mode=True)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_left, dummy_right, iters=iters, test_mode=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'fps': 1.0 / times.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=32)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--precision_dtype', default='float32')
    parser.add_argument('--mixed_precision', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Inference Benchmark: {args.height}x{args.width}, {args.iters} GRU iters")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print("=" * 70)

    results = []
    for name, module_name, ckpt_path in MODELS:
        print(f"\n[Loading] {name} ...")
        model = load_model(module_name, ckpt_path, args)

        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

        print(f"  Benchmarking FPS ...")
        fps_result = benchmark_fps(model, H=args.height, W=args.width,
                                   iters=args.iters, warmup=args.warmup,
                                   repeats=args.repeats)
        print(f"  Latency: {fps_result['mean_ms']:.1f} ± {fps_result['std_ms']:.1f} ms")
        print(f"  FPS: {fps_result['fps']:.2f}")

        results.append({
            'name': name,
            'params_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6,
            'latency_ms': fps_result['mean_ms'],
            'std_ms': fps_result['std_ms'],
            'fps': fps_result['fps'],
        })

        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Model':<28} {'Params(M)':<12} {'Latency(ms)':<14} {'FPS':<8}"
    print(header)
    print("-" * 62)
    for r in results:
        print(f"{r['name']:<28} {r['params_M']:<12.2f} {r['latency_ms']:<14.1f} {r['fps']:<8.2f}")

    # Diff
    if len(results) == 2:
        diff_params = results[1]['params_M'] - results[0]['params_M']
        diff_fps = results[1]['fps'] - results[0]['fps']
        print(f"\nΔ (B - A): Params {diff_params:+.4f}M, FPS {diff_fps:+.2f}")
        print("disparityrender adds only 1 learnable scalar (density_temperature).")

    print("=" * 70)


if __name__ == '__main__':
    main()
