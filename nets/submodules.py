from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .deform_conv import DeformConv, ModulatedDeformConv
import numpy as np


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False, padding_mode='zeros'),
                         nn.BatchNorm3d(out_planes))


class disparityrender(nn.Module):
    def __init__(self, mindisp=0, maxdisp=192, dpnum=10):
        super(disparityrender, self).__init__()
        stride = (maxdisp - mindisp) / dpnum
        self.register_buffer('disp', torch.linspace(maxdisp, mindisp, dpnum).view(1, -1, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        alpha = 1. - torch.exp(-x)
        alpha = torch.flip(alpha, [1])
        T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1, alpha.shape[2], alpha.shape[3])).to(alpha.device), 1. - alpha + 1e-10], 1), 1)[:, :-1, :, :]
        weights = alpha * T
        out = torch.sum(weights * self.disp.to(x.device), 1, keepdim=True)
        out = torch.squeeze(out, 1)
        return out


def conv1x1(in_planes, out_planes, groups=1, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class costtransdc_2cd(nn.Module):
    def __init__(self, dispnum):
        super(costtransdc_2cd, self).__init__()
        self.dispnum = dispnum

    def forward(self, Volumn):
        size = Volumn.size()
        Volumn = Volumn.view(size[0], self.dispnum, -1, size[-2], size[-1])
        Volumn = Volumn.permute(0, 2, 1, 3, 4).contiguous()
        Volumn = Volumn.view(size[0], -1, size[-2], size[-1])
        return Volumn.contiguous()


class costtranscd_2dc(nn.Module):
    def __init__(self, dispnum):
        super(costtranscd_2dc, self).__init__()
        self.dispnum = dispnum

    def forward(self, Volumn):
        size = Volumn.size()
        Volumn = Volumn.view(size[0], -1, self.dispnum, size[-2], size[-1])
        Volumn = Volumn.permute(0, 2, 1, 3, 4).contiguous()
        Volumn = Volumn.view(size[0], -1, size[-2], size[-1])
        return Volumn.contiguous()


class DeformConv2d(nn.Module):
    """A single (modulated) deformable conv layer"""

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=2,
                 groups=1,
                 deformable_groups=2,
                 modulation=True,
                 double_mask=True,
                 bias=False):
        super(DeformConv2d, self).__init__()

        self.modulation = modulation
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        self.double_mask = double_mask

        if self.modulation:
            self.deform_conv = ModulatedDeformConv(in_channels,
                                                   out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=dilation,
                                                   dilation=dilation,
                                                   groups=groups,
                                                   deformable_groups=deformable_groups,
                                                   bias=bias)
        else:
            self.deform_conv = DeformConv(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          groups=groups,
                                          deformable_groups=deformable_groups,
                                          bias=bias)

        k = 3 if self.modulation else 2

        offset_out_channels = deformable_groups * k * kernel_size * kernel_size

        self.offset_conv = nn.Conv2d(in_channels, offset_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=dilation, dilation=dilation,
                                     groups=deformable_groups, bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        if self.modulation:
            offset_mask = self.offset_conv(x)

            offset_channel = self.deformable_groups * 2 * self.kernel_size * self.kernel_size
            offset = offset_mask[:, :offset_channel, :, :]

            mask = offset_mask[:, offset_channel:, :, :]
            mask = mask.sigmoid()

            if self.double_mask:
                mask = mask * 2

            out = self.deform_conv(x, offset, mask)
        else:
            offset = self.offset_conv(x)
            out = self.deform_conv(x, offset)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False, leaky_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             relu)
    return conv


class DeformBottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, deformable_groups=1,
                 base_width=64, dilation=2, norm_layer=None, with_bn3=False):
        super(DeformBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * 1
        self.conv1 = conv1x1(inplanes, width, groups=deformable_groups)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride, dilation=dilation, groups=deformable_groups, deformable_groups=deformable_groups,
                 modulation=True,
                 double_mask=True)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, groups=deformable_groups)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.bn3D = None
        if(with_bn3):
            self.bn3D = BN23(groups)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if(self.bn3D is not None):
            out = self.bn3D(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, with_bn3=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * 1
            self.conv1 = conv1x1(inplanes, width, groups=groups)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion, groups=groups)
            self.bn3 = norm_layer((planes * self.expansion))
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            self.downsample = downsample
            self.stride = stride
        else:
            width = int(planes * (base_width / 64.)) * 1
            self.conv1 = conv1x1(inplanes, width, groups=groups)
            if norm_layer is BN23:
                self.bn1 = norm_layer(groups)
                self.bn2 = norm_layer(groups)
                self.bn3 = norm_layer(groups)
            else:
                self.bn1 = norm_layer(width // groups)
                if(width // groups > 1):
                    self.bn2 = norm_layer(width // groups // 2)
                else:
                    self.bn2 = norm_layer(width // groups)
                self.bn3 = norm_layer((planes * self.expansion) // groups)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.conv3 = conv1x1(width, planes * self.expansion, groups=groups)
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            self.downsample = downsample
            self.stride = stride
            if(with_bn3):
                self.bn3D = BN23t(groups)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
