"""Generate filenames for Middlebury v3 and ETH3D datasets.

Middlebury v3 directory structure:
    data_dir/
        MiddEval3/
            trainingH/        (or trainingF, trainingQ for different resolutions)
                scene_name/
                    im0.png
                    im1.png
                    disp0GT.pfm

ETH3D directory structure:
    data_dir/
        two_view_training/
            scene_name/
                im0.png
                im1.png
                disp0GT.pfm
        two_view_test/
            scene_name/
                im0.png
                im1.png

Usage:
    python generate_middlebury_eth3d.py --data_dir /path/to/data --dataset middlebury
    python generate_middlebury_eth3d.py --data_dir /path/to/data --dataset eth3d
"""

import os
import argparse


def generate_middlebury(data_dir, resolution='H'):
    """Generate filenames for Middlebury v3."""
    train_dir = os.path.join(data_dir, 'MiddEval3', f'training{resolution}')
    test_dir = os.path.join(data_dir, 'MiddEval3', f'test{resolution}')

    lines = []
    for split_dir, has_gt in [(train_dir, True), (test_dir, False)]:
        if not os.path.exists(split_dir):
            print(f'Warning: {split_dir} not found, skipping')
            continue
        scenes = sorted(os.listdir(split_dir))
        for scene in scenes:
            scene_path = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            rel_prefix = os.path.relpath(scene_path, data_dir)
            left = f'{rel_prefix}/im0.png'
            right = f'{rel_prefix}/im1.png'
            if has_gt:
                gt = f'{rel_prefix}/disp0GT.pfm'
                lines.append(f'{left} {right} {gt}')
            else:
                lines.append(f'{left} {right}')

    # Write train (with GT) and test (all) files
    train_lines = [l for l in lines if 'disp0GT' in l]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, 'Middlebury_train.txt'), 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    with open(os.path.join(out_dir, 'Middlebury_test.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Middlebury: {len(train_lines)} train, {len(lines)} total')


def generate_eth3d(data_dir):
    """Generate filenames for ETH3D two-view stereo."""
    train_dir = os.path.join(data_dir, 'two_view_training')
    test_dir = os.path.join(data_dir, 'two_view_test')

    lines = []
    for split_dir, has_gt in [(train_dir, True), (test_dir, False)]:
        if not os.path.exists(split_dir):
            print(f'Warning: {split_dir} not found, skipping')
            continue
        scenes = sorted(os.listdir(split_dir))
        for scene in scenes:
            scene_path = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            rel_prefix = os.path.relpath(scene_path, data_dir)
            left = f'{rel_prefix}/im0.png'
            right = f'{rel_prefix}/im1.png'
            if has_gt:
                gt = f'{rel_prefix}/disp0GT.pfm'
                lines.append(f'{left} {right} {gt}')
            else:
                lines.append(f'{left} {right}')

    train_lines = [l for l in lines if 'disp0GT' in l]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, 'ETH3D_train.txt'), 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    with open(os.path.join(out_dir, 'ETH3D_test.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'ETH3D: {len(train_lines)} train, {len(lines)} total')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--dataset', required=True, choices=['middlebury', 'eth3d'])
    parser.add_argument('--resolution', default='H', type=str,
                        help='Middlebury resolution: H(alf), F(ull), Q(uarter)')
    args = parser.parse_args()

    if args.dataset == 'middlebury':
        generate_middlebury(args.data_dir, args.resolution)
    else:
        generate_eth3d(args.data_dir)
