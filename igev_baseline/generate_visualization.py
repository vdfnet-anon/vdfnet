"""
Generate the Fig.2 generalization visualization for the paper.
Purpose: compare Exp A (soft-argmin) vs Exp B (disparityrender) on zero-shot
         disparity predictions over ETH3D/KITTI/Middlebury.

Usage:
    # Step 1: run inference on the server and save disparity maps (npy format)
    python generate_visualization.py --mode infer \
        --ckpt_a checkpoints/200000_igev_original.pth \
        --ckpt_b checkpoints/v6_igev_render/200000_igev_render.pth \
        --output_dir vis_results/

    # Step 2: compose the figure locally (no GPU needed)
    python generate_visualization.py --mode compose \
        --input_dir vis_results/ \
        --output fig2_generalization.png

Server paths:
    - ETH3D: /data/ETH3D/two_view_training/
    - KITTI: /data/KITTI/KITTI_2015/training/
    - Middlebury: /data/Middlebury/trainingH/
"""
import os
import sys
import argparse
import numpy as np
import importlib
from pathlib import Path

sys.path.append('core')


# ============================================================
# Config: which image to use for visualization per dataset
# Selection rule: pick the sample with the largest, most visually obvious difference
# ============================================================
SELECTED_SAMPLES = {
    'eth3d': {
        'scene': 'delivery_area_2l',  # Top 1: EPE 0.31→0.13, +57.3%
        'left': '/data/ETH3D/two_view_training/delivery_area_2l/im0.png',
        'right': '/data/ETH3D/two_view_training/delivery_area_2l/im1.png',
        'gt': '/data/ETH3D/two_view_training_gt/delivery_area_2l/disp0GT.pfm',
    },
    'kitti': {
        'scene': '000029_10',  # KITTI 2015 index 29: EPE 2.40→1.02, +57.4%
        'left': '/data/KITTI/KITTI_2015/training/image_2/000029_10.png',
        'right': '/data/KITTI/KITTI_2015/training/image_3/000029_10.png',
        'gt': '/data/KITTI/KITTI_2015/training/disp_occ_0/000029_10.png',
    },
    'middlebury': {
        'scene': 'Piano',  # Top 1: EPE 0.98→0.65, +33.9%
        'left': '/data/Middlebury/trainingH/Piano/im0.png',
        'right': '/data/Middlebury/trainingH/Piano/im1.png',
        'gt': '/data/Middlebury/trainingH/Piano/disp0GT.pfm',
    },
}


def read_pfm(filename):
    """Read a PFM-format disparity map"""
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = f.readline().decode('ascii').strip().split()
        width, height = int(dim_match[0]), int(dim_match[1])
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data


def load_gt(gt_path):
    """Load a ground truth disparity map"""
    if gt_path.endswith('.pfm'):
        disp = read_pfm(gt_path)
        valid = (disp > 0) & (disp < 192) & np.isfinite(disp)
    elif gt_path.endswith('.png'):
        disp = np.array(Image.open(gt_path)).astype(np.float32) / 256.0
        valid = disp > 0
    else:
        raise ValueError(f"Unknown GT format: {gt_path}")
    return disp, valid


def load_model(module_name, ckpt_path, args):
    """Load a model"""
    import torch
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


def run_inference(model, left_path, right_path, iters=32):
    """Run inference on a single image pair, returns the disparity map as a numpy array"""
    import torch
    from PIL import Image as PILImage
    from core.utils.utils import InputPadder

    image1 = np.array(PILImage.open(left_path)).astype(np.uint8)
    image2 = np.array(PILImage.open(right_path)).astype(np.uint8)

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].cuda()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].cuda()

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        disp_pr = model(image1, image2, iters=iters, test_mode=True)
    disp_pr = padder.unpad(disp_pr.float()).cpu().squeeze().numpy()

    return disp_pr


def mode_infer(args):
    """Step 1: run inference on the server"""
    import torch
    from PIL import Image as PILImage

    os.makedirs(args.output_dir, exist_ok=True)

    parser_model = argparse.Namespace(
        corr_levels=2, corr_radius=4, n_downsample=2,
        n_gru_layers=3, hidden_dims=[128, 128, 128],
        max_disp=192, precision_dtype='float32',
        mixed_precision=False,
    )

    models = {
        'exp_a': ('igev_stereo_original', args.ckpt_a),
        'exp_b': ('igev_stereo', args.ckpt_b),
    }

    for model_key, (module_name, ckpt_path) in models.items():
        print(f"\nLoading {model_key}: {ckpt_path}")
        model = load_model(module_name, ckpt_path, parser_model)

        for dataset_key, sample in SELECTED_SAMPLES.items():
            print(f"  Inferring {dataset_key}/{sample['scene']}...")

            if not os.path.exists(sample['left']):
                print(f"    SKIP: {sample['left']} not found")
                continue

            disp = run_inference(model, sample['left'], sample['right'], iters=32)

            save_path = os.path.join(args.output_dir, f"{dataset_key}_{model_key}.npy")
            np.save(save_path, disp)
            print(f"    Saved: {save_path} (shape={disp.shape})")

        del model
        torch.cuda.empty_cache()

    # Also save the left image and GT
    for dataset_key, sample in SELECTED_SAMPLES.items():
        if not os.path.exists(sample['left']):
            continue

        left_img = np.array(PILImage.open(sample['left']))
        np.save(os.path.join(args.output_dir, f"{dataset_key}_left.npy"), left_img)

        gt_path = sample['gt']
        if os.path.exists(gt_path):
            gt_disp, gt_valid = load_gt(gt_path)
            np.save(os.path.join(args.output_dir, f"{dataset_key}_gt.npy"), gt_disp)
            np.save(os.path.join(args.output_dir, f"{dataset_key}_valid.npy"), gt_valid)

    print(f"\nDone! Results saved to {args.output_dir}")
    print("Transfer to local machine and run: python generate_visualization.py --mode compose")


def mode_compose(args):
    """Step 2: compose the figure locally"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    input_dir = Path(args.input_dir)
    datasets = ['eth3d', 'kitti', 'middlebury']
    dataset_labels = ['ETH3D', 'KITTI 2015', 'Middlebury H']

    # Check that the files exist
    for ds in datasets:
        required = [f"{ds}_left.npy", f"{ds}_exp_a.npy", f"{ds}_exp_b.npy"]
        for f in required:
            if not (input_dir / f).exists():
                print(f"Missing: {input_dir / f}")
                print("Run --mode infer on server first.")
                return

    has_raft = all((input_dir / f"{ds}_raft_stereo.npy").exists() for ds in datasets)
    if not has_raft:
        print("Warning: RAFT-Stereo results not found, using 4-column layout (no RAFT-Stereo)")
        print("Run infer_raft_stereo.py on server to generate RAFT-Stereo predictions.")

    # 5 columns: Left | GT | RAFT-Stereo | IGEV (soft-argmin) | VDFNet (Ours)
    if has_raft:
        ncols = 5
        col_titles = ['Left Image', 'Ground Truth', 'RAFT-Stereo', 'IGEV-Stereo', 'VDFNet (Ours)']
    else:
        ncols = 4
        col_titles = ['Left Image', 'Ground Truth', 'IGEV-Stereo (soft-argmin)', 'VDFNet (Ours)']

    fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols, 8))

    for row, (ds, label) in enumerate(zip(datasets, dataset_labels)):
        left = np.load(input_dir / f"{ds}_left.npy")
        gt = np.load(input_dir / f"{ds}_gt.npy") if (input_dir / f"{ds}_gt.npy").exists() else None
        disp_a = np.load(input_dir / f"{ds}_exp_a.npy")
        disp_b = np.load(input_dir / f"{ds}_exp_b.npy")
        valid = np.load(input_dir / f"{ds}_valid.npy") if (input_dir / f"{ds}_valid.npy").exists() else None
        disp_raft = np.load(input_dir / f"{ds}_raft_stereo.npy") if has_raft else None

        # Determine colormap range
        if gt is not None:
            gt_safe = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
            if valid is None:
                valid = (gt_safe > 0) & (gt_safe < 192)
            vmin, vmax = 0, np.percentile(gt_safe[valid], 95) if valid.any() else gt_safe.max()
        else:
            vmax = max(np.percentile(disp_a, 95), np.percentile(disp_b, 95))
            vmin = 0

        col = 0

        # Col 0: Left image
        axes[row, col].imshow(left)
        axes[row, col].set_ylabel(label, fontsize=13, fontweight='bold')
        col += 1

        # Col 1: Ground truth
        if gt is not None:
            gt_vis = gt_safe.copy()
            gt_vis[~valid] = 0
            axes[row, col].imshow(gt_vis, cmap='magma', vmin=vmin, vmax=vmax)
        else:
            axes[row, col].set_facecolor('black')
        col += 1

        # Col 2: RAFT-Stereo (if available)
        if has_raft:
            axes[row, col].imshow(disp_raft, cmap='magma', vmin=vmin, vmax=vmax)
            col += 1

        # Col 3: IGEV (soft-argmin) = Exp A
        axes[row, col].imshow(disp_a, cmap='magma', vmin=vmin, vmax=vmax)
        col += 1

        # Col 4: VDFNet (disparityrender) = Exp B
        axes[row, col].imshow(disp_b, cmap='magma', vmin=vmin, vmax=vmax)

        # Remove ticks
        for c in range(ncols):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')

    plt.tight_layout(pad=0.5)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output}")
    plt.close()


def mode_select(args):
    """Helper mode: iterate over all samples in a dataset, compute the per-image EPE difference, to help pick the best visualization sample"""
    import torch
    from PIL import Image as PILImage

    parser_model = argparse.Namespace(
        corr_levels=2, corr_radius=4, n_downsample=2,
        n_gru_layers=3, hidden_dims=[128, 128, 128],
        max_disp=192, precision_dtype='float32',
        mixed_precision=False,
    )

    print("Loading models...")
    model_a = load_model('igev_stereo_original', args.ckpt_a, parser_model)
    model_b = load_model('igev_stereo', args.ckpt_b, parser_model)

    dataset_name = args.dataset
    print(f"\nScanning {dataset_name} for best visualization samples...")

    if dataset_name == 'eth3d':
        import core.stereo_datasets as datasets
        val_dataset = datasets.ETH3D({})
    elif dataset_name == 'kitti':
        import core.stereo_datasets as datasets
        val_dataset = datasets.KITTI({}, image_set='training')
    elif dataset_name == 'middlebury':
        import core.stereo_datasets as datasets
        val_dataset = datasets.Middlebury({}, split='H')

    results = []
    for val_id in range(len(val_dataset)):
        if dataset_name == 'eth3d':
            (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        elif dataset_name == 'middlebury':
            (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        else:
            _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            imageL_file = f"sample_{val_id:06d}"

        image1_cuda = image1[None].cuda()
        image2_cuda = image2[None].cuda()

        from core.utils.utils import InputPadder
        padder = InputPadder(image1_cuda.shape, divis_by=32)
        img1_pad, img2_pad = padder.pad(image1_cuda, image2_cuda)

        with torch.no_grad():
            disp_a = model_a(img1_pad, img2_pad, iters=32, test_mode=True)
            disp_b = model_b(img1_pad, img2_pad, iters=32, test_mode=True)

        disp_a = padder.unpad(disp_a.float()).cpu().squeeze()
        disp_b = padder.unpad(disp_b.float()).cpu().squeeze()

        val = (valid_gt.squeeze() >= 0.5) & (flow_gt.squeeze().abs() < 192)
        gt = flow_gt.squeeze()

        epe_a = (disp_a - gt).abs()[val].mean().item()
        epe_b = (disp_b - gt).abs()[val].mean().item()
        improvement = (epe_a - epe_b) / epe_a * 100

        results.append((imageL_file, epe_a, epe_b, improvement))
        print(f"  [{val_id:3d}] {Path(imageL_file).parent.name:20s} "
              f"EPE_A={epe_a:.4f} EPE_B={epe_b:.4f} Δ={improvement:+.1f}%")

    # Sort by improvement magnitude
    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n{'='*60}")
    print(f"Top 5 samples where Exp B is MOST better than Exp A:")
    for i, (name, ea, eb, imp) in enumerate(results[:5]):
        print(f"  {i+1}. {Path(name).parent.name}: EPE {ea:.4f} → {eb:.4f} ({imp:+.1f}%)")


if __name__ == '__main__':
    from PIL import Image

    parser = argparse.ArgumentParser(description='Generate Fig.2 visualization for VDFNet paper')
    parser.add_argument('--mode', required=True, choices=['infer', 'compose', 'select'],
                        help='infer=run on server, compose=make figure locally, select=find best samples')
    parser.add_argument('--ckpt_a', default='checkpoints/200000_igev_original.pth',
                        help='Exp A checkpoint path')
    parser.add_argument('--ckpt_b', default='checkpoints/v6_igev_render/200000_igev_render.pth',
                        help='Exp B checkpoint path')
    parser.add_argument('--output_dir', default='vis_results/',
                        help='Directory to save inference results')
    parser.add_argument('--input_dir', default='vis_results/',
                        help='Directory with inference results (for compose mode)')
    parser.add_argument('--output', default='fig2_generalization.png',
                        help='Output figure path (for compose mode)')
    parser.add_argument('--dataset', default='eth3d', choices=['eth3d', 'kitti', 'middlebury'],
                        help='Dataset to scan (for select mode)')
    args = parser.parse_args()

    if args.mode == 'infer':
        mode_infer(args)
    elif args.mode == 'compose':
        mode_compose(args)
    elif args.mode == 'select':
        mode_select(args)
