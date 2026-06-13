"""
FLOPs and parameter count for VDFNet.

Usage:
    python count_flops.py                        # default: 384x1248 (KITTI)
    python count_flops.py --height 540 --width 960
    python count_flops.py --height 256 --width 512  # SceneFlow training size
"""
import argparse
import torch
from thop import profile
from utils.utils import count_parameters
import nets

parser = argparse.ArgumentParser()
parser.add_argument('--height', type=int, default=384, help='Input image height')
parser.add_argument('--width',  type=int, default=1248, help='Input image width')
args = parser.parse_args()

# Pad to multiple of 16 (same as predict.py)
import math
factor = 16
H = math.ceil(args.height / factor) * factor
W = math.ceil(args.width  / factor) * factor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nets.stereorf(args).to(device)
model.eval()

left  = torch.randn(1, 3, H, W).to(device)
right = torch.randn(1, 3, H, W).to(device)

flops, _ = profile(model, inputs=(left, right), verbose=False)
params = count_parameters(model)

print(f'Input size : {H} x {W}')
print(f'FLOPs      : {flops / 1e9:.2f} G')
print(f'Params     : {params / 1e6:.2f} M')