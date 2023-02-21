import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import BasicBlock, Bottleneck
import math
from math import sqrt

def conv2d(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding  = (kernel_size//2), bias = bias)

#before batchnorm bias should false
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias = False)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
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


# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(ChannelAttention, self).__init__()
        
        #get point from global average
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # get features 
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, padding = 0, bias = True)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 =  nn.Conv2d(channel // reduction, channel, 1, padding = 0, bias = True)
        self.sigmoid = nn.Sigmoid()    


        # nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu") 
  
        # nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="sigmoid")   
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class ResBlockAttention(nn.Module):
    def __init__(self, conv, features, kernel_size, reduction, batch_norm = False, res_scale = 1): #batch_norm true
        super(ResBlockAttention, self).__init__()

        Reslayers = []
        conv2d = conv(features, features, kernel_size, bias = True)
        batch_norm = nn.BatchNorm2d(features)
        for i in range(2):
            Reslayers.append(conv2d)
            if batch_norm:
                Reslayers.append(batch_norm)
            if i == 0:
                Reslayers.append(nn.ReLU(inplace = True))
        
        Reslayers.append(ChannelAttention(features))

        self.layers = nn.Sequential(*Reslayers)

        self.res_scale  = res_scale

        # nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu") 
        # nn.init.constant_(batch_norm.weight, 1)
        # nn.init.constant_(batch_norm.bias, 0)

    def forward(self, x):
        res = x
        out = self.layers(x)

        out += res

        return out

class ResGroup(nn.Module):
    def __init__(self, conv, features, kernel_size, reduction, res_scale, n_resblocks):
        super(ResGroup, self).__init__()

        Reslayers = []
        Reslayers = [
            ResBlockAttention(conv, features, kernel_size, reduction, batch_norm = True) #batch_norm true
            for _ in range(n_resblocks)
        ]

        Reslayers.append(conv(features, features, kernel_size))
        Reslayers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*Reslayers)

    def forward(self, x):
        res = x
        out = self.layers(x)

        out += res
        return out
        

class Network(nn.Module):
    def __init__(self,features, n_resgroup, n_resblocks, scale = 1 ,conv = conv2d, block = BasicBlock):
        super(Network, self).__init__()
        
        " old network: sử dụng một lớp pooling rồi cho qua khối res in res, xong đó up pooling rồi concat x với pooling vừa xong cho qua softmax"
        '''
        This network use before BasicBlock 1 2 3 with concat downsample and upsample
        Use basic block with downsample residual 
        Let check some reslock to see different of accuracy
        Note for train: bias all = True, set đầu ra là conv2d với kenel là 1, rồi sử dụng batchnorm
        Try to
        '''
        kernel_size = 3
        res_scale = 1
        reduction = 16
        # scale = 8
        
        self.downsample1 = nn.Sequential(
            conv1x1(features, features,1),
            nn.BatchNorm2d(features)
        )
        self.block1 = block(features, features, downsample = self.downsample1, init_weight = False)
        self.maxpool = nn.MaxPool2d(scale)
        
        grouplayers = [
            ResGroup(conv, features, kernel_size, reduction = reduction, res_scale = res_scale, n_resblocks = n_resblocks)
            for _ in range(n_resgroup)
        ]
        self.layers = nn.Sequential(*grouplayers)

        self.downsample2 = nn.Sequential(
            conv1x1(features*2, features,1),
            nn.BatchNorm2d(features)
        )

        # self.uppool = up_pooling(features, features) # up theo paper and concat lại
        self.uppool = Upsampler(conv, scale, features, act  = False)
        self.block2 = block(features*2, features, downsample = self.downsample2, init_weight = False)
        # self.conv2 = nn.Conv2d(in_channels = features*2, out_channels = features, kernel_size=1, padding=0, bias = True )
       

        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # cat = x
        cat = self.block1(x) # 128 56 56
        out = self.maxpool(cat)     
        res = self.layers(out)#128 28 28
        res += out

        out = self.uppool(res)#64 56 56
        out = torch.cat([out, cat], dim = 1) #192
        out = self.block2(out) #128 ---> 64
        # out = self.conv2(out)

        out = torch.softmax(out, dim = 1)
        return out
        



class Networklast(nn.Module):
    def __init__(self, features, n_resgroup, n_resblocks, conv = conv2d, block = BasicBlock):
        super(Networklast, self).__init__()

        kernel_size = 3
        reduction = 16
        scale = 1
        res_scale =1 

        self.downsample1 = nn.Sequential(
            conv1x1(features, features,1),
            nn.BatchNorm2d(features)
        )
        self.block1 = block(features, features, downsample = self.downsample1, init_weight = False)

        # self.maxpool = nn.MaxPool2d(2)
        grouplayers = [
            ResGroup(conv, features, kernel_size, reduction = reduction, res_scale = res_scale, n_resblocks = n_resblocks)
            for _ in range(n_resgroup)
        ]
        self.layers = nn.Sequential(*grouplayers)
        self.downsample2 = nn.Sequential(
            conv1x1(features, features,1),
            nn.BatchNorm2d(features)
        )
        # self.uppool = Upsampler(conv, scale, features, act  = False)
        self.block2 = block(features, features, downsample = self.downsample2, init_weight = False)
        
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
 

    def forward(self, x):
        out = self.block1(x)
        # out = self.maxpool(cat)
        out = self.layers(out)
        # out = self.uppool(out)
        # out = torch.cat([out, cat], dim = 1)
        out = self.block2(out)

        out = torch.softmax(out, dim = 1)
        return out