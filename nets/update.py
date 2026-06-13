import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *

class Groupfus_conv(nn.Module):# no channel reduction in deform bottleneck; the bottleneck replaces grouped convolution
    def __init__(self, initalchannel,groups,dilation=1,useconvfilter=False,usedeformconvfuse=True,convfinalfuse=False,with_SA=False):
        super(Groupfus_conv, self).__init__()
        self.initalc=initalchannel
        self.dpnum=initalchannel//groups
        dilationn=2 #             fixed the fitting bug 20221101; stride-2 dilated convolution
        convmodual=[]
        fc=initalchannel
        convgroup=groups
        dilationn=dilationn*2

        for i in range(1):
            convmodual.append(nn.Sequential(
                DeformBottleneck(inplanes=fc, planes=fc, stride=1, dilation=dilation, deformable_groups=convgroup,base_width=64),  
                costtranscd_2dc(self.dpnum),
                Bottleneck(inplanes=fc, planes=fc, stride=1, dilation=1, groups=self.dpnum,base_width=32,norm_layer=None), 
                 costtransdc_2cd(self.dpnum)
                 ))
            
        self.convmodual = nn.Sequential(*convmodual)
    def forward(self, Volumn):
        return self.convmodual(Volumn) 
    
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-1, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, groups, corr_dim=72, hidden_dim=128, input_dim=128, outflow=False,withinp=True, withflow=False, mask_size=4):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = Groupfus_conv(corr_dim, groups)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.outflow = outflow
        self.inp = withinp
        self.withflow = withflow
        
        if self.withflow:
            self.convf1 = nn.Conv2d(1, 32, 7, padding=3)
            self.convf2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv = nn.Conv2d(32+corr_dim, hidden_dim-1, 3, padding=1)
        if self.outflow:
            self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
            self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, mask_size**2 * 9, 1, padding=0),
        )


    def forward(self, net, inp, corr,flow=None):
        corr = self.encoder(corr)
        if self.withflow:
            flo = self.convf1(flow)
            flo = self.convf2(flo)
            cor_flo = torch.cat([corr, flo], dim=1)
            out = F.relu(self.conv(cor_flo))
            corr=torch.cat([out, flow], dim=1)
        if self.inp:
            inp = torch.cat([inp, corr], dim=1)
        else:
            inp = corr
        net = self.gru(net, inp)
        if self.outflow:
            delta_flow = self.flow_head(net)
            mask = 0.25 * self.mask(net)
            return net, delta_flow, mask
        else:
            return net



