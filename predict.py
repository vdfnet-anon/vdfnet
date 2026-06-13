import torch
import torch.nn.functional as F
import cv2
import skimage.io
import argparse
import numpy as np
import os
import math

import nets
import time
from dataloader import transforms
from utils import utils
from utils.file_io import write_pfm, read_img
from glob import glob
from numpy import savez_compressed
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data_dir', default=None, required=True, type=str, help='Data directory for prediction')
parser.add_argument('--file_pattern', default=None, type=str,
                    help='Glob pattern for left images (e.g. "*/im0.png" for Middlebury/ETH3D). '
                         'If not set, auto-detect based on directory structure.')
parser.add_argument('--right_pattern', default=None, type=str,
                    help='How to derive right image path from left. Options: '
                         '"left2right" (replace left with right), '
                         '"im0_im1" (replace im0.png with im1.png). '
                         'If not set, auto-detect.')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--output_dir', default=None, type=str,
                    help='Directory to save inference results. If not set, uses data_dir/pred/')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--pretrained_vdfnet', default=None, type=str, help='Pretrained network')
parser.add_argument('--model', default='stereorf', choices=['stereorf', 'stereorf_gwc', 'stereorf_v2'], help='Model architecture')

# Save
parser.add_argument('--save_type', default='png', choices=['pfm', 'png', 'npy', 'npz'], help='Save file type')
parser.add_argument('--visualize', action='store_true', help='Save color visualization')
parser.add_argument('--save_dir', default='pred', type=str, help='Save prediction directory')

args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = os.path.join(args.data_dir, args.save_dir)
utils.check_path(args.output_dir)


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range < 1e-8:
        return np.zeros_like(data)
    return (data - np.min(data)) / _range


def find_samples(data_dir, file_pattern=None, right_pattern=None):
    """Auto-detect or use specified patterns to find stereo image pairs."""
    data_dir = data_dir.rstrip('/')

    # Auto-detect dataset structure
    if file_pattern is None:
        # KITTI style: left/*.png, right/*.png
        kitti_samples = sorted(glob(os.path.join(data_dir, 'left', '*.png')))
        if kitti_samples:
            return kitti_samples, 'left2right'
        # Also try image_2/image_3 (KITTI benchmark)
        kitti_bench = sorted(glob(os.path.join(data_dir, 'image_2', '*.png')))
        if kitti_bench:
            return kitti_bench, 'image2_image3'
        # Middlebury/ETH3D style: scene_name/im0.png
        mb_samples = sorted(glob(os.path.join(data_dir, '*/im0.png')))
        if mb_samples:
            return mb_samples, 'im0_im1'
        # Middlebury nested: resolution/scene_name/im0.png
        mb_nested = sorted(glob(os.path.join(data_dir, '*/*/im0.png')))
        if mb_nested:
            return mb_nested, 'im0_im1'
        raise FileNotFoundError(f'No stereo pairs found in {data_dir}. '
                                f'Use --file_pattern to specify.')
    else:
        samples = sorted(glob(os.path.join(data_dir, file_pattern)))
        if not samples:
            raise FileNotFoundError(f'No files matching {file_pattern} in {data_dir}')
        if right_pattern is None:
            right_pattern = 'im0_im1'
        return samples, right_pattern


def get_right_path(left_path, pattern):
    """Derive right image path from left image path."""
    if pattern == 'left2right':
        return left_path.replace('/left/', '/right/')
    elif pattern == 'image2_image3':
        return left_path.replace('image_2', 'image_3')
    elif pattern == 'im0_im1':
        return left_path.replace('im0.png', 'im1.png')
    else:
        raise ValueError(f'Unknown right_pattern: {pattern}')

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    # Load model
    if args.model == 'stereorf_gwc':
        model = nets.stereorf_gwc(args).to(device)
    elif args.model == 'stereorf_v2':
        model = nets.stereorf_v2(args).to(device)
    else:
        model = nets.stereorf(args).to(device)
    if args.pretrained_vdfnet and os.path.exists(args.pretrained_vdfnet):
        print('=> Loading pretrained model:', args.pretrained_vdfnet)
        utils.load_pretrained_net(model, args.pretrained_vdfnet, no_strict=True)
    else:
        print('=> Warning: using random initialization')

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.eval()

    # Find stereo pairs
    all_samples, right_pattern = find_samples(args.data_dir, args.file_pattern, args.right_pattern)
    num_samples = len(all_samples)
    print('=> %d samples found in %s' % (num_samples, args.data_dir))

    avg_time = 0
    avg_mem = 0
    factor = 16  # padding alignment factor

    for i, left_name in enumerate(all_samples):
        if i % 100 == 0:
            print('=> Inferencing %d/%d' % (i, num_samples))

        right_name = get_right_path(left_name, right_pattern)
        if not os.path.exists(right_name):
            print(f'Warning: right image not found: {right_name}, skipping')
            continue

        left = read_img(left_name)
        right = read_img(right_name)
        sample = {'left': left, 'right': right}
        sample = test_transform(sample)

        left = sample['left'].to(device).unsqueeze(0)   # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)

        # Auto padding to multiple of factor
        ori_height, ori_width = left.size()[2:]
        pad_height = math.ceil(ori_height / factor) * factor
        pad_width = math.ceil(ori_width / factor) * factor
        top_pad = pad_height - ori_height
        right_pad = pad_width - ori_width

        if top_pad > 0 or right_pad > 0:
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            start_time = time.time()
            pred_disp = model(left, right)[-1]  # [B, H, W]
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            if i > 0:
                avg_time += elapsed

        pred_disp = pred_disp.squeeze(1)
        if pred_disp.size(-1) < left.size(-1):
            pred_disp = pred_disp.unsqueeze(1)
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                      mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)

        # Crop padding
        if top_pad > 0 or right_pad > 0:
            if right_pad > 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        disp = pred_disp[0].detach().cpu().numpy()  # [H, W]

        if i > 0:
            avg_mem += torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        # Save disparity
        # For Middlebury/ETH3D, use scene_name/im0 to avoid overwriting
        rel_path = os.path.relpath(left_name, args.data_dir)
        rel_stem = rel_path[:-4]  # strip .png
        save_name = os.path.join(args.output_dir, rel_stem + '.png')
        utils.check_path(os.path.dirname(save_name))

        if args.save_type == 'pfm':
            if args.visualize:
                skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))
            save_name = save_name[:-3] + 'pfm'
            write_pfm(save_name, disp)
        elif args.save_type == 'npy':
            save_name = save_name[:-3] + 'npy'
            np.save(save_name, disp)
        elif args.save_type == 'npz':
            save_name = save_name[:-3] + 'npz'
            savez_compressed(save_name, disp)
        else:
            skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))

        # Save color visualization
        if args.visualize:
            color_dir = os.path.join(args.output_dir, 'color')
            utils.check_path(os.path.join(color_dir, os.path.dirname(rel_stem)))
            color_name = os.path.join(color_dir, rel_stem + '.png')
            disp_vis = normalization(disp) * 255.
            disp_vis = 255. - disp_vis
            im_color = cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_JET)
            Image.fromarray(im_color).save(color_name)

    if num_samples > 1:
        print("avg_time: %.4f s" % (avg_time / (num_samples - 1)))
        print("avg_mem: %.2f GB" % (avg_mem / (num_samples - 1)))


if __name__ == '__main__':
    main()
