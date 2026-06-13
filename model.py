import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import os
import numpy as np

from utils import utils
from utils.visualization import disp_error_img, save_images
from metric import d1_metric, thres_metric


class Model(object):
    def __init__(self, args, logger, optimizer, vdfnet, device, start_iter=0, start_epoch=0,
                 best_epe=None, best_epoch=None, rank=0):
        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.vdfnet = vdfnet
        self.device = device
        self.num_iter = start_iter
        self.epoch = start_epoch
        self.rank = rank

        self.best_epe = 999. if best_epe is None else best_epe
        self.best_epoch = -1 if best_epoch is None else best_epoch

        if not args.evaluate_only and rank == 0:
            self.train_writer = SummaryWriter(self.args.checkpoint_dir)

        self.use_amp = getattr(args, 'amp', False)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)

    def train(self, train_loader):
        args = self.args
        logger = self.logger

        steps_per_epoch = len(train_loader)
        device = self.device

        self.vdfnet.train()

        if args.freeze_bn:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.vdfnet.apply(set_bn_eval)

        last_print_time = time.time()

        for i, sample in enumerate(train_loader):
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]
            mask = (gt_disp > 0) & (gt_disp < args.max_disp)
            if args.load_pseudo_gt:
                pseudo_gt_disp = sample['pseudo_disp'].to(device)
                pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < args.max_disp) & (~mask)  # inverse mask
            if not mask.any():
                continue

            with autocast('cuda', enabled=self.use_amp):
                pred_disp_pyramid = self.vdfnet(left, right)
                sampleh, samplew = left.size(2) // 4, left.size(3) // 4

                gt_disp_fine = F.interpolate(gt_disp.unsqueeze(1), size=(sampleh, samplew), mode='nearest').squeeze(1)
                maskfine = (gt_disp_fine > 0) & (gt_disp_fine < args.max_disp)
                if not maskfine.any():
                    continue

                disp_loss = 0
                pseudo_disp_loss = 0
                pyramid_loss = []
                pseudo_pyramid_loss = []
                # Loss weights
                if len(pred_disp_pyramid) == 5:
                    pyramid_weight = [0.6, 0.8, 1.0, 1.2, 1.4]
                elif len(pred_disp_pyramid) == 4:
                    pyramid_weight = [0.125, 0.25, 0.5, 1.0]
                elif len(pred_disp_pyramid) == 3:
                    pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
                elif len(pred_disp_pyramid) == 2:
                    pyramid_weight = [0.5, 1.0]  # highest loss only
                elif len(pred_disp_pyramid) == 1:
                    pyramid_weight = [1.0]  # highest loss only
                else:
                    raise NotImplementedError

                assert len(pyramid_weight) == len(pred_disp_pyramid)
                for k in range(len(pred_disp_pyramid)):
                    pred_disp = pred_disp_pyramid[k]
                    weight = pyramid_weight[k]
                    pred_disp = pred_disp.squeeze(1)

                    if pred_disp.size(-1) != gt_disp.size(-1):
                        curr_loss = F.smooth_l1_loss(pred_disp[maskfine], gt_disp_fine[maskfine],
                                                 reduction='mean')
                    else:
                        curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                                 reduction='mean')
                    disp_loss += weight * curr_loss
                    pyramid_loss.append(curr_loss)

                    # Pseudo gt loss
                    if args.load_pseudo_gt:
                        pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_gt_disp[pseudo_mask],
                                                            reduction='mean')
                        pseudo_disp_loss += weight * pseudo_curr_loss

                        pseudo_pyramid_loss.append(pseudo_curr_loss)

                total_loss = disp_loss

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.num_iter += 1

            if self.num_iter % args.print_freq == 0 and self.rank == 0:
                this_cycle = time.time() - last_print_time
                last_print_time += this_cycle

                time_per_step = this_cycle / args.print_freq
                remaining_steps = (args.max_epoch - self.epoch) * steps_per_epoch - (i + 1)
                eta_seconds = remaining_steps * time_per_step
                eta_h, eta_m = int(eta_seconds // 3600), int((eta_seconds % 3600) // 60)

                logger.info('Epoch: [%3d/%3d] [%5d/%5d] time: %4.2fs disp_loss: %.3f ETA: %dh%02dm' %
                            (self.epoch + 1, args.max_epoch, i + 1, steps_per_epoch, this_cycle,
                             disp_loss.item(), eta_h, eta_m))

        self.epoch += 1

        # Always save the latest model for resuming training
        if args.no_validate and self.rank == 0:
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.vdfnet,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  epe=-1, best_epe=self.best_epe,
                                  best_epoch=self.best_epoch,
                                  filename='vdfnet_latest.pth')

            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, 'models')
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.vdfnet,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=-1, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      save_optimizer=False)

    def validate(self, val_loader):
        args = self.args
        logger = self.logger
        if self.rank == 0:
            logger.info('=> Start validation...')

        if args.evaluate_only is True:
            if args.pretrained_vdfnet is not None:
                pretrained_vdfnet = args.pretrained_vdfnet
            else:
                model_name = 'vdfnet_best.pth'
                pretrained_vdfnet = os.path.join(args.checkpoint_dir, model_name)
                if not os.path.exists(pretrained_vdfnet):
                    pretrained_vdfnet = pretrained_vdfnet.replace(model_name, 'vdfnet_latest.pth')

            logger.info('=> loading pretrained vdfnet: %s' % pretrained_vdfnet)
            utils.load_pretrained_net(self.vdfnet, pretrained_vdfnet, no_strict=True)

        self.vdfnet.eval()

        num_samples = len(val_loader)
        if self.rank == 0:
            logger.info('=> %d samples found in the validation set' % num_samples)

        val_epe = torch.zeros(1, device=self.device)
        val_d1 = torch.zeros(1, device=self.device)
        val_thres1 = torch.zeros(1, device=self.device)
        val_thres2 = torch.zeros(1, device=self.device)
        val_thres3 = torch.zeros(1, device=self.device)

        val_count = 0

        val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')

        num_imgs = 0
        valid_samples = 0

        for i, sample in enumerate(val_loader):
            if i % 100 == 0 and self.rank == 0:
                logger.info('=> Validating %d/%d' % (i, num_samples))

            left = sample['left'].to(self.device)  # [B, 3, H, W]
            right = sample['right'].to(self.device)
            gt_disp = sample['disp'].to(self.device)  # [B, H, W]
            mask = (gt_disp > 0) & (gt_disp < args.max_disp)
            if not mask.any():
                continue

            valid_samples += 1

            num_imgs += gt_disp.size(0)

            with torch.no_grad():
                pred_disp = self.vdfnet(left, right)[-1]

            if pred_disp.size(-1) < gt_disp.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                          mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]
            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

            val_epe += epe
            val_d1 += d1
            val_thres1 += thres1
            val_thres2 += thres2
            val_thres3 += thres3

            # Save 3 images for visualization
            if not args.evaluate_only and self.rank == 0:
                if i in [num_samples // 4, num_samples // 2, num_samples // 4 * 3]:
                    img_summary = dict()
                    img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                    img_summary['left'] = left
                    img_summary['right'] = right
                    img_summary['gt_disp'] = gt_disp
                    img_summary['pred_disp'] = pred_disp
                    save_images(self.train_writer, 'val' + str(val_count), img_summary, self.epoch)
                    val_count += 1

        if self.rank == 0:
            logger.info('=> Validation done!')

        mean_epe = (val_epe / valid_samples).item()
        mean_d1 = (val_d1 / valid_samples).item()
        mean_thres1 = (val_thres1 / valid_samples).item()
        mean_thres2 = (val_thres2 / valid_samples).item()
        mean_thres3 = (val_thres3 / valid_samples).item()

        if self.rank != 0:
            return

        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % self.epoch)
            f.write('epe: %.3f\t' % mean_epe)
            f.write('d1: %.4f\t' % mean_d1)
            f.write('thres1: %.4f\t' % mean_thres1)
            f.write('thres2: %.4f\t' % mean_thres2)
            f.write('thres3: %.4f\n' % mean_thres3)

        logger.info('=> Mean validation epe of epoch %d: %.3f' % (self.epoch, mean_epe))

        if not args.evaluate_only:
            self.train_writer.add_scalar('val/epe', mean_epe, self.epoch)
            self.train_writer.add_scalar('val/d1', mean_d1, self.epoch)
            self.train_writer.add_scalar('val/thres1', mean_thres1, self.epoch)
            self.train_writer.add_scalar('val/thres2', mean_thres2, self.epoch)
            self.train_writer.add_scalar('val/thres3', mean_thres3, self.epoch)

        if not args.evaluate_only:
            if args.val_metric == 'd1':
                if mean_d1 < self.best_epe:
                    # Actually best_epe here is d1
                    self.best_epe = mean_d1
                    self.best_epoch = self.epoch

                    utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.vdfnet,
                                          epoch=self.epoch, num_iter=self.num_iter,
                                          epe=mean_d1, best_epe=self.best_epe,
                                          best_epoch=self.best_epoch,
                                          filename='vdfnet_best.pth')
            elif args.val_metric == 'epe':
                if mean_epe < self.best_epe:
                    self.best_epe = mean_epe
                    self.best_epoch = self.epoch

                    utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.vdfnet,
                                          epoch=self.epoch, num_iter=self.num_iter,
                                          epe=mean_epe, best_epe=self.best_epe,
                                          best_epoch=self.best_epoch,
                                          filename='vdfnet_best.pth')
            else:
                raise NotImplementedError

        if self.epoch == args.max_epoch:
            # Save best validation results
            with open(val_file, 'a') as f:
                f.write('\nbest epoch: %03d \t best %s: %.3f\n\n' % (self.best_epoch,
                                                                     args.val_metric,
                                                                     self.best_epe))

            logger.info('=> best epoch: %03d \t best %s: %.3f\n' % (self.best_epoch,
                                                                    args.val_metric,
                                                                    self.best_epe))

        if not args.evaluate_only:
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.vdfnet,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  epe=mean_epe, best_epe=self.best_epe,
                                  best_epoch=self.best_epoch,
                                  filename='vdfnet_latest.pth')

            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, 'models')
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.vdfnet,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=mean_epe, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      save_optimizer=False)
