import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
import numpy as np
import os

import nets
import dataloader
from dataloader import transforms
from utils import utils
import model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')

# Training data
parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=64, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=288, type=int, help='Image height for training')
parser.add_argument('--img_width', default=512, type=int, help='Image width for training')

# For KITTI, using 384x1248 for validation
parser.add_argument('--val_img_height', default=576, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=960, type=int, help='Image width for validation')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--checkpoint_dir', default=None, type=str, required=True,
                    help='Directory to save model checkpoints and logs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_epoch', default=64, type=int, help='Maximum epoch number for training')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

parser.add_argument('--pretrained_vdfnet', default=None, type=str, help='Pretrained network')
parser.add_argument('--freeze_bn', action='store_true', help='Switch BN to eval mode to fix running statistics')

# Learning rate
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma')
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', help='Type of learning rate scheduler')
parser.add_argument('--milestones', default=None, type=str, help='Milestones for MultiStepLR')

# Loss
parser.add_argument('--highest_loss_only', action='store_true', help='Only use loss on highest scale for finetuning')
parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

# Log
parser.add_argument('--print_freq', default=100, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (epochs)')

parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')
parser.add_argument('--no_validate', action='store_true', help='No validation')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')
parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
parser.add_argument('--model', default='stereorf', choices=['stereorf', 'stereorf_gwc', 'stereorf_v2'], help='Model architecture')
parser.add_argument('--grad_accum_steps', default=1, type=int, help='Gradient accumulation steps')

args = parser.parse_args()

logger = utils.get_logger()


def main():
    # DDP setup. When launched without torchrun (no LOCAL_RANK in env), fall
    # back to single-GPU/CPU training so the script is runnable out of the box.
    ddp = 'LOCAL_RANK' in os.environ
    if ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('=> LOCAL_RANK not set: running in single-process mode on %s '
                    '(use torchrun for multi-GPU DDP)' % device)

    if rank == 0:
        utils.check_path(args.checkpoint_dir)
        utils.save_args(args)
        filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
        utils.save_command(args.checkpoint_dir, filename)

    if ddp:
        dist.barrier()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # Train loader
    train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                            transforms.RandomColor(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
    train_transform = transforms.Compose(train_transform_list)

    train_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                          dataset_name=args.dataset_name,
                                          mode='train' if args.mode != 'train_all' else 'train_all',
                                          load_pseudo_gt=args.load_pseudo_gt,
                                          transform=train_transform)

    if rank == 0:
        logger.info('=> {} training samples found in the training set'.format(len(train_data)))

    train_sampler = DistributedSampler(train_data, shuffle=True) if ddp else None
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=4)

    # Validation loader
    val_transform_list = [transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
    val_transform = transforms.Compose(val_transform_list)
    val_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                        dataset_name=args.dataset_name,
                                        mode='val',
                                        transform=val_transform)

    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=True, prefetch_factor=4)

    # Network
    if args.model == 'stereorf_gwc':
        vdfnet = nets.stereorf_gwc(args).to(device)
    elif args.model == 'stereorf_v2':
        vdfnet = nets.stereorf_v2(args).to(device)
    else:
        vdfnet = nets.stereorf(args).to(device)
    if rank == 0:
        logger.info('%s' % vdfnet)

    if args.pretrained_vdfnet is not None:
        if rank == 0:
            logger.info('=> Loading pretrained VDFNet: %s' % args.pretrained_vdfnet)
        utils.load_pretrained_net(vdfnet, args.pretrained_vdfnet, no_strict=(not args.strict))

    if ddp:
        vdfnet = DDP(vdfnet, device_ids=[local_rank], output_device=local_rank)

    # Save parameters
    num_params = utils.count_parameters(vdfnet)
    if rank == 0:
        world_size = dist.get_world_size() if ddp else 1
        logger.info('=> Use %d GPU(s) (%s)' % (world_size, 'DDP' if ddp else 'single'))
        logger.info('=> Number of trainable parameters: %d' % num_params)
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.Adam(vdfnet.parameters(), args.learning_rate, betas=(0.9, 0.999))

    # Resume training
    if args.resume:
        start_epoch, start_iter, best_epe, best_epoch = utils.resume_latest_ckpt(
            args.checkpoint_dir, vdfnet, 'vdfnet')

        # Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, 'optimizer')
    else:
        start_epoch = 0
        start_iter = 0
        best_epe = None
        best_epoch = None

    # LR scheduler
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)
        elif args.lr_scheduler_type == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=args.max_epoch,
                                                                       eta_min=1e-5,
                                                                       last_epoch=last_epoch)
        else:
            raise NotImplementedError

    train_model = model.Model(args, logger, optimizer, vdfnet, device, start_iter, start_epoch,
                              best_epe=best_epe, best_epoch=best_epoch, rank=rank)

    if rank == 0:
        logger.info('=> Start training...')

    if args.evaluate_only:
        assert args.val_batch_size == 1
        train_model.validate(val_loader)
    else:
        for epoch in range(start_epoch, args.max_epoch):
            if ddp:
                train_sampler.set_epoch(epoch)
            if not args.evaluate_only:
                train_model.train(train_loader)
            if not args.no_validate:
                train_model.validate(val_loader)
            if args.lr_scheduler_type is not None:
                lr_scheduler.step()

        if rank == 0:
            logger.info('=> End training\n\n')

    if ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
