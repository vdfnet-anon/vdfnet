"""
Run RAFT-Stereo inference on the GPU server to produce the disparity maps
needed for the generalization visualizations.

Prerequisites:
    1. Clone RAFT-Stereo: git clone https://github.com/princeton-vl/RAFT-Stereo.git
    2. Download the SceneFlow pretrained weights:
       from https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J
       download raftstereo-sceneflow.pth
    3. Install dependencies: pip install timm

Usage:
    python infer_raft_stereo.py \
        --raft_stereo_dir $WORKSPACE/RAFT-Stereo \
        --checkpoint $WORKSPACE/RAFT-Stereo/models/raftstereo-sceneflow.pth \
        --output_dir vis_results/
"""
import os
import sys
import argparse
import numpy as np


SELECTED_SAMPLES = {
    'eth3d': {
        'scene': 'delivery_area_2l',
        'left': '/data/ETH3D/two_view_training/delivery_area_2l/im0.png',
        'right': '/data/ETH3D/two_view_training/delivery_area_2l/im1.png',
    },
    'kitti': {
        'scene': '000029_10',
        'left': '/data/KITTI/KITTI_2015/training/image_2/000029_10.png',
        'right': '/data/KITTI/KITTI_2015/training/image_3/000029_10.png',
    },
    'middlebury': {
        'scene': 'Piano',
        'left': '/data/Middlebury/trainingH/Piano/im0.png',
        'right': '/data/Middlebury/trainingH/Piano/im1.png',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raft_stereo_dir', required=True, help='Path to RAFT-Stereo repo')
    parser.add_argument('--checkpoint', required=True, help='Path to raftstereo-sceneflow.pth')
    parser.add_argument('--output_dir', default='vis_results/')
    args = parser.parse_args()

    import torch
    sys.path.insert(0, args.raft_stereo_dir)
    from core.raft_stereo import RAFTStereo
    from core.utils.utils import InputPadder

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model_args = argparse.Namespace(
        hidden_dims=[128, 128, 128],
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm='batch',
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=False,
        shared_backbone=False,
        corr_implementation='reg',
    )

    model = RAFTStereo(model_args)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # Strip the module. prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    print(f"RAFT-Stereo loaded: {args.checkpoint}")

    from PIL import Image

    for dataset_key, sample in SELECTED_SAMPLES.items():
        left_path = sample['left']
        right_path = sample['right']

        if not os.path.exists(left_path):
            print(f"  SKIP: {left_path} not found")
            continue

        print(f"  Inferring {dataset_key}/{sample['scene']}...")

        image1 = np.array(Image.open(left_path)).astype(np.uint8)
        image2 = np.array(Image.open(right_path)).astype(np.uint8)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].cuda()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            _, disp_pr = model(image1, image2, iters=32, test_mode=True)

        disp_pr = -disp_pr  # RAFT-Stereo outputs negative disparity
        disp_pr = padder.unpad(disp_pr.float()).cpu().squeeze().numpy()

        save_path = os.path.join(args.output_dir, f"{dataset_key}_raft_stereo.npy")
        np.save(save_path, disp_pr)
        print(f"    Saved: {save_path} (shape={disp_pr.shape})")

    print("\nDone!")


if __name__ == '__main__':
    main()
