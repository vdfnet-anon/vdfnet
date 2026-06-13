"""
Orin NX inference-speed benchmark.
Measures VDFNet (disparityrender) inference time and memory usage on Jetson Orin NX.

Usage:
    python3 benchmark_orin.py \
        --ckpt checkpoints/igev_render_ft4.pth \
        --model igev_stereo \
        --width 640 --height 480 \
        --warmup 10 --repeats 50
"""
import os
import sys
import time
import argparse
import importlib
import numpy as np
import torch

sys.path.append('core')


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


def benchmark(model, height, width, iters, warmup, repeats, use_fp16=False):
    image1 = torch.randn(1, 3, height, width).cuda()
    image2 = torch.randn(1, 3, height, width).cuda()

    # padding to multiple of 32
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32
    if pad_h > 0 or pad_w > 0:
        image1 = torch.nn.functional.pad(image1, [0, pad_w, 0, pad_h])
        image2 = torch.nn.functional.pad(image2, [0, pad_w, 0, pad_h])

    if use_fp16:
        image1 = image1.float()  # keep float32, autocast handles conversion
        image2 = image2.float()

    ctx = torch.amp.autocast('cuda', enabled=use_fp16, dtype=torch.float16)

    # warmup
    print(f"  Warming up ({warmup} iters)...")
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            _ = model(image1, image2, iters=iters, test_mode=True)
    torch.cuda.synchronize()

    # benchmark
    print(f"  Benchmarking ({repeats} iters)...")
    times = []
    with torch.no_grad(), ctx:
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(image1, image2, iters=iters, test_mode=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = np.array(times)
    return times.mean(), times.std()


def get_memory_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model', default='igev_stereo',
                        choices=['igev_stereo', 'igev_stereo_original'])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--iters', type=int, default=32)
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

    print(f"\n{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Input size: {args.height}x{args.width}")
    print(f"GRU iters:  {args.iters}")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()

    print("\nLoading model...")
    model = load_model(args.model, args.ckpt, args)
    print("Using FP16 autocast" if args.mixed_precision else "Using FP32")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params/1e6:.2f}M")

    mean_ms, std_ms = benchmark(model, args.height, args.width,
                                args.iters, args.warmup, args.repeats,
                                use_fp16=args.mixed_precision)
    mem_mb = get_memory_mb()

    print(f"\n{'='*60}")
    print(f"Latency:    {mean_ms:.1f} ± {std_ms:.1f} ms")
    print(f"FPS:        {1000/mean_ms:.2f}")
    print(f"Memory:     {mem_mb/1024:.2f} GB ({mem_mb:.0f} MB)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
