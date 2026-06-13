from __future__ import print_function, division
import sys
sys.path.append('core')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from igev_stereo import IGEVStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        image_epe = epe_flattened[val].mean().item()
        epe_list.append(image_epe)
        out_list.append({
            'bad05': (epe_flattened[val] > 0.5).float().mean().item(),
            'bad10': (epe_flattened[val] > 1.0).float().mean().item(),
            'bad20': (epe_flattened[val] > 2.0).float().mean().item(),
        })
        logging.info(f"ETH3D {val_id+1}/{len(val_dataset)}. EPE {image_epe:.4f}")

    epe = np.mean(epe_list)
    bad05 = 100 * np.mean([x['bad05'] for x in out_list])
    bad10 = 100 * np.mean([x['bad10'] for x in out_list])
    bad20 = 100 * np.mean([x['bad20'] for x in out_list])

    print(f"Validation ETH3D: EPE {epe:.4f}, bad-0.5 {bad05:.2f}%, bad-1.0 {bad10:.2f}%, bad-2.0 {bad20:.2f}%")
    return {'eth3d-epe': epe, 'eth3d-bad05': bad05, 'eth3d-bad10': bad10, 'eth3d-bad20': bad20}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        image_epe = epe_flattened[val].mean().item()
        epe_list.append(image_epe)
        out_list.append(epe_flattened[val].cpu().numpy())
        if val_id < 9 or (val_id+1) % 10 == 0:
            logging.info(f"KITTI {val_id+1}/{len(val_dataset)}. EPE {image_epe:.4f}. Runtime: {end-start:.3f}s")

    epe_list = np.array(epe_list)
    out_all = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1_all  = 100 * np.mean(out_all > 3.0)
    bad_2   = 100 * np.mean(out_all > 2.0)
    bad_4   = 100 * np.mean(out_all > 4.0)

    avg_runtime = np.mean(elapsed_list)
    print(f"Validation KITTI: EPE {epe:.4f}, D1-all(>3px) {d1_all:.2f}%, bad-2 {bad_2:.2f}%, bad-4 {bad_4:.2f}%, {1/avg_runtime:.2f}-FPS")
    return {'kitti-epe': epe, 'kitti-d1': d1_all, 'kitti-bad2': bad_2, 'kitti-bad4': bad_4}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe = torch.abs(flow_pr - flow_gt)

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if(np.isnan(epe[val].mean().item())):
            continue

        epe_list.append(epe[val].mean().item())
        out_list.append(epe[val].cpu().numpy())
        # if val_id == 400:
        #     break

    epe_list = np.array(epe_list)
    out_all = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1  = 100 * np.mean(out_all > 3.0)
    er1 = 100 * np.mean(out_all > 1.0)

    f = open('test.txt', 'a')
    f.write("Validation Scene Flow: EPE %f, 1-ER %f, 3-ER %f\n" % (epe, er1, d1))

    print("Validation Scene Flow: EPE %.4f, 1-ER %.2f%%, 3-ER %.2f%%" % (epe, er1, d1))
    return {'scene-disp-epe': epe, 'scene-disp-1er': er1, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask == 255)
        image_epe = epe_flattened[val].mean().item()
        epe_list.append(image_epe)
        out_list.append(epe_flattened[val].cpu().numpy())
        logging.info(f"Middlebury {val_id+1}/{len(val_dataset)}. EPE {image_epe:.4f}")

    epe = np.mean(epe_list)
    out_all = np.concatenate(out_list)
    bad_20 = 100 * np.mean(out_all > 2.0)
    bad_40 = 100 * np.mean(out_all > 4.0)

    print(f"Validation Middlebury{split}: EPE {epe:.4f}, bad-2.0 {bad_20:.2f}%, bad-4.0 {bad_40:.2f}%")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-bad20': bad_20, f'middlebury{split}-bad40': bad_40}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)
