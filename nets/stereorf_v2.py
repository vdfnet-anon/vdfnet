from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from .submodules import *
from .update import *

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes*2)) # +conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes)) # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)
        return out, pre, post


class MLP_full(nn.Module):
    def __init__(self, num_pt, num_ch, scale):
        super(MLP_full, self).__init__()
        self.in_ch = num_ch
        self.num_pt = num_pt
        self.scale = scale
        self.W = 64
        self.pts_bias = nn.Conv3d(self.in_ch, self.W, 1, 1, 0, 1, bias=True)
        self.pts1 = nn.Conv3d(self.num_pt, self.W, 1, 1, 0, 1, bias=True)
        self.pts2 = nn.Conv3d(self.W, self.W, 1, 1, 0, 1, bias=True)
        self.bn2 = nn.BatchNorm3d(self.W)
        self.pts5 = nn.Conv3d(self.W + self.num_pt, self.W, 1, 1, 0, 1, bias=True)
        self.pts6 = nn.Conv3d(self.W, self.W, 1, 1, 0, 1, bias=True)
        self.out = nn.Conv3d(self.W, 1, 1, 1, 0, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, points, feature):
        bias = self.pts_bias(feature)
        pt = self.relu(self.pts1(points) + bias)
        pt = self.relu(self.bn2(self.pts2(pt) + bias))
        pt = self.relu(self.pts5(torch.cat((pt, points), dim=1)) + bias)
        pt = self.relu(self.pts6(pt) + bias)
        pt = self.out(pt)
        return self.relu(pt)


class stereorf_v2(nn.Module):

    def __init__(self, arg):
        super(stereorf_v2, self).__init__()
        
        self.maxdisp = 192

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.bandlength = 8
        self.bandendlist = 8
        self.bandstartlist = 2
        self.fineserchr = 4  # 5090 32GB only fits fineserchr=4 (6/8/12 all OOM)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True), nn.ConvTranspose3d(32, 1, 4, 4, 0, bias=True), nn.ReLU(inplace=True))
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True), nn.ConvTranspose3d(32, 1, 4, 4, 0, bias=True), nn.ReLU(inplace=True))
        self.classif3 = MLP_full(6 * self.bandlength, 32, 4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def cost_sample(self, cost, disp, smpw, smph, smpdr):
        size = cost.size()
        dev = cost.device
        B = size[0]
        ndr = 2 * smpdr + 1

        freq_bands = (2 ** torch.linspace(self.bandstartlist, self.bandendlist, self.bandlength, device=dev)
                      ).view(1, self.bandlength, 1, 1, 1).expand(B, self.bandlength, ndr, smph, smpw)

        offset_x = 0
        offset_y = 0

        Dr = torch.linspace(-smpdr, smpdr, ndr, device=dev).view(1, ndr, 1, 1).expand(B, ndr, smph, smpw)

        Dc = disp[:, offset_y:offset_y + smph, offset_x:offset_x + smpw].unsqueeze(1).expand(B, ndr, smph, smpw) + Dr
        Dc = Dc / 192.  # normalize before PE for resolution-invariant encoding
        d = Dc.unsqueeze(1).expand(B, self.bandlength, ndr, smph, smpw) * freq_bands

        Uc = (torch.linspace(offset_x, offset_x + smpw - 1, smpw, device=dev).view(1, 1, 1, smpw).expand(B, ndr, smph, smpw)
              / (size[4] * 4))
        u = Uc.unsqueeze(1).expand(B, self.bandlength, ndr, smph, smpw) * freq_bands

        Vc = (torch.linspace(offset_y, offset_y + smph - 1, smph, device=dev).view(1, 1, smph, 1).expand(B, ndr, smph, smpw)
              / (size[3] * 4))
        v = Vc.unsqueeze(1).expand(B, self.bandlength, ndr, smph, smpw) * freq_bands
        
        candgrid = torch.stack((Dc, Vc, Uc), dim=4)
        cost_sample = F.grid_sample(cost, candgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        x = torch.cat((u, v, d), 1)
        # 3D positional encoding, order: u, v, d
        candibase = torch.cat((torch.sin(x), torch.cos(x)), 1)
        if self.training:
            return cost_sample, candibase
        else:
            return cost_sample, candibase

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # Matching with concat method (vectorized, no Python loop)
        B, C, H, W = refimg_fea.size()
        D = self.maxdisp // 4
        cost = torch.zeros(B, C * 2, D, H, W, device=refimg_fea.device, dtype=refimg_fea.dtype)
        cost[:, :C, 0, :, :] = refimg_fea
        cost[:, C:, 0, :, :] = targetimg_fea
        for i in range(1, D):
            cost[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            cost[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        if self.training:
            out1, pre1, post1 = checkpoint(self.dres2, cost0, None, None, use_reentrant=False)
            out1 = out1 + cost0
            out2, pre2, post2 = checkpoint(self.dres3, out1, pre1, post1, use_reentrant=False)
            out2 = out2 + cost0
            out3, pre3, post3 = checkpoint(self.dres4, out2, pre1, post2, use_reentrant=False)
            out3 = out3 + cost0
        else:
            out1, pre1, post1 = self.dres2(cost0, None, None)
            out1 = out1 + cost0
            out2, pre2, post2 = self.dres3(out1, pre1, post1)
            out2 = out2 + cost0
            out3, pre3, post3 = self.dres4(out2, pre1, post2)
            out3 = out3 + cost0

        tx = 0
        ty = 0
        sw = 0
        sh = 0
        if self.training:
            rf1 = self.classif1(out1)
            pred1 = disparityrender(0, self.maxdisp, self.maxdisp)(rf1.squeeze(1))
            rf2 = self.classif2(out2)
            pred2 = disparityrender(0, self.maxdisp, self.maxdisp)(rf2.squeeze(1))
            sw = pred2.size()[2]
            sh = pred2.size()[1]
            wrapcost, candibase = self.cost_sample(out3, pred2, sw, sh, self.fineserchr)
            rf3 = checkpoint(self.classif3, candibase, wrapcost, use_reentrant=False)
            pred3 = disparityrender(-self.fineserchr, self.fineserchr, 2 * self.fineserchr + 1)(rf3.squeeze(1)).squeeze(1) + disparityrender(1, 1, 2 * self.fineserchr + 1)(rf3.squeeze(1)).squeeze(1) * pred2[:, ty:ty + sh, tx:tx + sw].detach()
        else:
            rf1 = self.classif1(out1)
            pred1 = disparityrender(0, self.maxdisp, self.maxdisp)(rf1.squeeze(1))
            rf2 = self.classif2(out2) 
            pred2 = disparityrender(0, self.maxdisp, self.maxdisp)(rf2.squeeze(1))
            wrapcost, candibase = self.cost_sample(out3, pred2, pred2.size()[2], pred2.size()[1], self.fineserchr)
            rf3 = self.classif3(candibase, wrapcost)
            pred3 = disparityrender(-self.fineserchr, self.fineserchr, 2 * self.fineserchr + 1)(rf3.squeeze(1)).squeeze(1) + disparityrender(1, 1, 2 * self.fineserchr + 1)(rf3.squeeze(1)).squeeze(1) * pred2

        if self.training:
            return [pred1, pred2, pred3]
        else:
            return [pred3]
