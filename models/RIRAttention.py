"""
Create block RIR BAM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import BasicBlock
import math
from math import sqrt
from .Attention import *
from .cbam import *

def conv2d(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size//2), bias = bias)

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2 , padding = 0, bias = True)
        self.pool = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
    def forward(self, x):
        out = torch.cat([self.conv(x), self.pool(x)], 1)
        out = self.bn(x)
        return F.relu(out)

def Up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, channels, bn = False, act = False, bias = True):
        
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(conv(channels, 4* channels, 3, bias))
                layers.append(nn.PixelShuffle(2))
                if bn:
                    layers.append(nn.BatchNorm2d(channels))
                if act:
                    layers.append(act())
        elif scale == 3:
            layers.append(conv(channels, 9 * channels, 3 , bias))
            layers.append(nn.PixelShuffle(3))
            if bn:
                layers.append(BatchNorm2d(channels))
            if act:
                layers.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*layers)

class ResBlockAttention(nn.Module):
    def __init__(self, conv, channels, kernel_size, reduction, batch_norm = False):
        super(ResBlockAttention, self).__init__()
        ResLayers = []
        
        self.downsample = nn.Sequential(
            conv1x1(channels, channels,1),
            nn.BatchNorm2d(channels)
        )

        self.block = BasicBlock(channels, channels, downsample = self.downsample, init_weight = True)

        ResLayers.append(CBAM(channels))
        self.layers = nn.Sequential(*ResLayers)

    def forward(self, x):
        res = x

        out = self.block(x)
        out = self.layers(out)
        out += res
 
        
        return out

class ResGroup(nn.Module):
    def __init__(self, conv, channels,kernel_size,reduction, n_resblocks):
        super(ResGroup, self).__init__()
        
        ResLayers = []
        ResLayers = [
            ResBlockAttention(conv, channels, kernel_size, reduction)
            for _ in range(n_resblocks)
        ]
                
        self.layers = nn.Sequential(*ResLayers)
        self.conv = nn.Conv2d(channels, channels, kernel_size = 1, padding = 0, bias = False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        out = self.layers(x)

        out += x
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

class Module1(nn.Module):
    def __init__(self, channels, n_resgroups, n_resblocks, scale, conv = conv2d):
        super(Module1, self).__init__()
        
        kernel_size = 3
        reduction = 16
        # size1 = (56,56)
        # size2 = (28,28)
        # size3 = (14,14)
        # size4 = (7,7)

        self.CBAM = CBAM(channels)
 
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        # self.down4 = nn.MaxPool2d(2)

        groupLayers = [
            ResGroup(conv, channels, kernel_size, reduction = reduction, n_resblocks = n_resblocks )
            for _ in range(n_resgroups)
        ]

        self.deepblock = nn.Sequential(*groupLayers)
        '''
        self.up1 = nn.UpsamplingBilinear2d(size = size1)
        self.up2 = nn.UpsamplingBilinear2d(size = size2)
        self.up3 = nn.UpsamplingBilinear2d(size = size3)
        self.up4 = nn.UpsamplingBilinear2d(size = size4)
        '''

        self.up1 = Up_pooling(channels, channels)
        self.up2 = Up_pooling(channels, channels)
        self.up3 = Up_pooling(channels, channels)
        # # self.up4 = Up_pooling(channels, channels)

        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride =1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels, channels, kernel_size=1, stride = 1, bias = False),
            nn.Sigmoid(),
        )
        #try to use this code and check accuracy
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out_cat = self.CBAM(x) #56
        
        out = self.down1(x)  #28
        out_deep1 = self.deepblock(out)
        out = self.down2(out_deep1)#14 
        out_deep2 = self.deepblock(out) 
        out = self.down3(out_deep2) #7
        out_deep3 = self.deepblock(out)
        # out = self.down4(out_deep3) #3.5
        # out_deep4 = self.deepblock(out)

        # out = self.up4(out_deep4)
        # out = out + out_deep3
        out = self.up3(out_deep3) #14
        out = out + out_deep2
        out = self.up2(out_deep2) #28
        out = out + out_deep1
        out = self.up1(out) # 56
        out = self.out(out)

        out = (1+out)*out_cat
        # out = torch.softmax(out, dim = 1)
        return out

class Module2(nn.Module):
    def __init__(self, channels, n_resgroups, n_resblocks, scale, conv = conv2d):
        super(Module2, self).__init__()
        
        kernel_size = 3
        reduction = 16
        # size1 = (28,28)
        # size2 = (14,14)
        # size3 = (7,7)

        self.CBAM = CBAM(channels)
 
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        # self.down3 = nn.MaxPool2d(2)

        groupLayers = [
            ResGroup(conv, channels, kernel_size, reduction = reduction, n_resblocks = n_resblocks )
            for _ in range(n_resgroups)
        ]

        self.deepblock = nn.Sequential(*groupLayers)

        self.up1 = Up_pooling(channels, channels)
        self.up2 = Up_pooling(channels, channels)
        # self.up1 = nn.UpsamplingBilinear2d(size = size1)
        # self.up2 = nn.UpsamplingBilinear2d(size = size2)
        # self.up3 = Up_pooling(channels, channels)
        # self.up4 = Up_pooling(channels, channels)

        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride =1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels, channels, kernel_size=1, stride = 1, bias = False),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        out_cat = self.CBAM(x)
        
        out = self.down1(x)  #28
        out_deep1 = self.deepblock(out)
        out = self.down2(out_deep1) #14
        out_deep2 = self.deepblock(out) 
        # out = self.down3(out_deep2)
        # out_deep3 = self.deepblock(out)

        # out = self.up3(out_deep3)
        # out = out + out_deep2
        out = self.up2(out_deep2) #28
        out = out + out_deep1
        out = self.up1(out) # 56
        out = self.out(out)

        out = (1+out)*out_cat
        # out = torch.softmax(out, dim = 1)

        return out

class Module3(nn.Module):
    def __init__(self, channels, n_resgroups, n_resblocks, scale, conv = conv2d):
        super(Module3, self).__init__()
        
        kernel_size = 3
        reduction = 16
        # size1 = (14,14)
        # size2 = (7,7)

        self.CBAM = CBAM(channels)
 
        self.down1 = nn.MaxPool2d(2)
        # self.down2 = nn.MaxPool2d(2)

        groupLayers = [
            ResGroup(conv, channels, kernel_size, reduction = reduction, n_resblocks = n_resblocks )
            for _ in range(n_resgroups)
        ]

        self.deepblock = nn.Sequential(*groupLayers)

        # self.up1 = nn.UpsamplingBilinear2d(size = size1)

        self.up1 = Up_pooling(channels, channels)
        # self.up2 = Up_pooling(channels, channels)

        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride =1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels, channels, kernel_size=1, stride = 1, bias = False),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        out_cat = self.CBAM(x)

        out = self.down1(x)
        out_deep1 = self.deepblock(out)
        # out = self.down2(out_deep1)
        # out_deep2 = self.deepblock(out)

        # out = self.up2(out_deep2)
        out = out + out_deep1
        out = self.up1(out)

        out = self.out(out)
        out = (1+out)*out_cat
        # out = torch.softmax(out, dim = 1)

        return out

class Module4(nn.Module):
    def __init__(self, channels, n_resgroups, n_resblocks, scale, conv = conv2d):
        super(Module4, self).__init__()
        
        kernel_size = 3
        reduction = 16
        # size1 = (14,14)
        # size2 = (7,7)

        self.CBAM = CBAM(channels)
 
        # self.down1 = nn.MaxPool2d(2)
        # self.down2 = nn.MaxPool2d(2)

        groupLayers = [
            ResGroup(conv, channels, kernel_size, reduction = reduction, n_resblocks = n_resblocks )
            for _ in range(n_resgroups)
        ]

        self.deepblock = nn.Sequential(*groupLayers)

        # self.up1 = nn.UpsamplingBilinear2d(size = size1)
        # self.up2 = nn.UpsamplingBilinear2d(size = size2)

        self.out = nn.Sequential(
            # nn.BatchNorm2d(channels),
            # nn.ReLU(inplace = True),
            nn.Conv2d(channels, channels, kernel_size=1, stride =1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels, channels, kernel_size=1, stride = 1, bias = False),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out_cat = self.CBAM(x)
        out = self.deepblock(x)
        
        out = self.out(out)
        out = (1+out)*out_cat
        # out = torch.softmax(out, dim = 1)


        return out
