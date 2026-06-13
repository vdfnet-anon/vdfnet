"""
Evaluate 3-ER on the SceneFlow test set using the native VDFNet code.
Works for PSMNet-VDF (stereorf) and GwcNet-VDF (stereorf_gwc).

Usage:
    export VDFNET_DATA=/path/to/SceneFlow
    python3 eval_vdfnet_3er.py --model stereorf --ckpt /path/to/vdfnet_best.pth
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Repo root = directory holding this script (so `nets` imports work from anywhere)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nets


def read_pfm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        color = header == b'PF'
        dim_match = f.readline().decode('ascii').strip().split()
        width, height = int(dim_match[0]), int(dim_match[1])
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data.astype(np.float32)


def get_test_samples(data_dir):
    samples = []
    things_dir = Path(data_dir) / 'FlyingThings3D'
    img_base = things_dir / 'frames_finalpass' / 'TEST'
    disp_base = things_dir / 'disparity' / 'TEST'
    for subset in sorted(img_base.iterdir()):
        for scene in sorted(subset.iterdir()):
            left_dir = scene / 'left'
            right_dir = scene / 'right'
            disp_dir = disp_base / subset.name / scene.name / 'left'
            if not left_dir.exists():
                continue
            for img_file in sorted(left_dir.glob('*.png')):
                right_file = right_dir / img_file.name
                disp_file = disp_dir / (img_file.stem + '.pfm')
                if right_file.exists() and disp_file.exists():
                    samples.append((str(img_file), str(right_file), str(disp_file)))
    return samples


def load_image(path):
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).float()


def pad_to(tensor, multiple=16):
    h, w = tensor.shape[-2], tensor.shape[-1]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph > 0 or pw > 0:
        tensor = F.pad(tensor, [0, pw, 0, ph])
    return tensor, ph, pw


@torch.no_grad()
def evaluate(model, samples, max_disp=192):
    model.cuda()
    model.eval()
    epe_list, er1_list, er3_list = [], [], []

    for left_path, right_path, disp_path in tqdm(samples):
        left = load_image(left_path).unsqueeze(0).cuda()
        right = load_image(right_path).unsqueeze(0).cuda()
        h_orig, w_orig = left.shape[-2], left.shape[-1]

        left_pad, ph, pw = pad_to(left, 16)
        right_pad, _, _ = pad_to(right, 16)

        output = model(left_pad, right_pad)
        if isinstance(output, (list, tuple)):
            disp_pred = output[-1]
        else:
            disp_pred = output

        if disp_pred.dim() == 4:
            disp_pred = disp_pred.squeeze(1)
        disp_pred = disp_pred[..., :h_orig, :w_orig].cpu().numpy().squeeze()

        disp_gt = read_pfm(disp_path)
        valid = (disp_gt > 0) & (disp_gt < max_disp) & np.isfinite(disp_gt)
        if not valid.any():
            continue

        err = np.abs(disp_pred - disp_gt)
        epe_list.append(err[valid].mean())
        er1_list.append((err[valid] > 1.0).mean() * 100)
        er3_list.append((err[valid] > 3.0).mean() * 100)

    print(f"\n{'='*50}")
    print(f"N:      {len(epe_list)}")
    print(f"EPE:    {np.mean(epe_list):.4f}")
    print(f"1-ER:   {np.mean(er1_list):.2f}%")
    print(f"3-ER:   {np.mean(er3_list):.2f}%")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='stereorf', choices=['stereorf', 'stereorf_gwc', 'stereorf_v2'])
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_dir', default=os.environ.get('VDFNET_DATA', 'data/SceneFlow'),
                        help='SceneFlow root (or set $VDFNET_DATA)')
    parser.add_argument('--maxdisp', type=int, default=192)
    args = parser.parse_args()

    print(f"Loading {args.model} from {args.ckpt}")

    model_args = argparse.Namespace(maxdisp=args.maxdisp)
    if args.model == 'stereorf_gwc':
        model = nets.stereorf_gwc(model_args).cuda()
    elif args.model == 'stereorf_v2':
        model = nets.stereorf_v2(model_args).cuda()
    else:
        model = nets.stereorf(model_args).cuda()

    state = torch.load(args.ckpt, map_location='cpu')
    if 'state_dict' in state:
        sd = state['state_dict']
    elif 'model' in state:
        sd = state['model']
    else:
        sd = state
    new_sd = {k.replace('module.', ''): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params/1e6:.2f}M")

    print("Collecting samples...")
    samples = get_test_samples(args.data_dir)
    print(f"Found {len(samples)} samples")

    evaluate(model, samples, args.maxdisp)


if __name__ == '__main__':
    main()
