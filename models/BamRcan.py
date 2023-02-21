import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import BasicBlock, Bottleneck
import math
from math import sqrt   
from .Bam import *

def conv2d(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size//2), bias = bias)

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size = 3, stride = 2 , padding = 1, bias = True)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.bn = nn.BatchNorm2d()
    def forward(self, x):
        out = torch.cat([self.conv(x), self.pool(x)], 1)
        out = self.bn(x)
        return F.relu(out)

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

class ResBlockBam(nn.Module):
    def __init__(self, conv, channels, kernel_size, reduction, batch_norm = False):
        super(ResBlockBam, self).__init__()

        ResLayers = []
        conv2d = conv(channels, channels, kernel_size, bias = False)
        batch_norm = nn.BatchNorm2d(channels)

        for i in range(2):
            ResLayers.append(conv2d)
            if batch_norm:
                ResLayers.append(batch_norm)
            if i == 0:
                ResLayers.append(nn.ReLU(inplace = True))

        ResLayers.append(BAM(channels, reduction))
        self.layers = nn.Sequential(*ResLayers)
        
    def forward(self, x):
        out = self.layers(x)

        out += x

        return out

class ResGroup(nn.Module):
    def __init__(self, conv, channels, kernel_size, reduction, n_resblocks):
        super(ResGroup, self).__init__()

        ResLayers = []
        ResLayers = [
            ResBlockBam(conv, channels, kernel_size , reduction, batch_norm = False)
            for _ in range(n_resblocks)
        ]

        ResLayers.append(conv(channels, channels, kernel_size))
        # ResLayers.append(nn.ReLU(inplace = True))
        # ResLayers.append(nn.ReLU(inplace = True))
        self.layers = nn.Sequential(*ResLayers)
        # self.relu_out = nn.ReLU(inplace = False)
        

    def forward(self, x):
        res = x
        out = self.layers(x)

        out += res

        # out = self.relu_out(out)

        return out


class BamNetwork(nn.Module):
    def __init__(self, channels, n_resgroup, n_resblocks, scale, conv = conv2d, block = BasicBlock):
        super(BamNetwork, self).__init__()

        kernel_size = 3
        reduction = 16
        # scale = 8
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)

        # self.downsample1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 1, stride = 2, padding = 1)

        
        grouplayers = [
            ResGroup(conv, channels, kernel_size, reduction = reduction, n_resblocks = n_resblocks)
            for _ in range(n_resgroup)
        ]
        self.layers = nn.Sequential(*grouplayers)

        # self.uppool = up_pooling(channels, channels) # up theo paper and concat lại
        self.uppool = Upsampler(conv, scale, channels, act  = False)
        # self.block2 = block(channels, channels, downsample = self.downsample2, init_weight = False)
        self.conv_out = nn.Conv2d(in_channels = channels*2, out_channels = channels, kernel_size = 1, padding = 0, stride = 1 )   
        self.relu_out = nn.ReLU(inplace = True) 

    def forward(self, x):

        max_out = self.maxpool(x)     
        avg_out = self.avgpool(x)

        res_max = self.layers(max_out)#128 28 28
        res_max += max_out

        res_avg = self.layers(avg_out)#128 28 28
        res_avg += avg_out

        up_max = self.uppool(res_max)#64 56 56
        up_avg = self.uppool(res_avg)
        out = torch.cat([up_max, up_avg], dim = 1)

        # out = torch.cat([out, x], dim = 1) #192
        out = self.conv_out(out) #128 ---> 64
        out = self.relu_out(out)

        out = torch.softmax(out, dim = 1)
        return out
        
    
class BamNetworklast(nn.Module):
    def __init__(self, channels, n_resgroup, n_resblocks, conv = conv2d, block = BasicBlock):
        super(BamNetworklast, self).__init__()

        kernel_size = 3
        reduction = 16
        scale = 1
         

        # self.maxpool = nn.MaxPool2d(2)
        grouplayers = [
            ResGroup(conv, channels, kernel_size, reduction = reduction, n_resblocks = n_resblocks)
            for _ in range(n_resgroup)
        ]
        self.layers = nn.Sequential(*grouplayers)
        self.conv_out = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 1, padding = 0, stride = 1 )   
        self.relu_out = nn.ReLU(inplace = True)

    def forward(self, x):
        # out = self.block1(x)
        # out = self.maxpool(cat)
        res = self.layers(x)
        res += x
        # out = self.uppool(out)
        # out = torch.cat([out, cat], dim = 1)
        out = self.conv_out(res)
        out = self.relu_out(out)
        out = torch.softmax(out, dim = 1)
        return out
