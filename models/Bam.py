import torch.nn as nn
import torch
import torch.nn.functional as F 
from .resnet import BasicBlock


#create Bam block

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(channels, channels // ratio, 1, bias = True)
        self.relu = nn.PReLU(channels // ratio)
        self.conv2 = nn.Conv2d(channels // ratio, channels, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)       
        out = self.sigmoid(out)
        
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7) , 'kernal size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size = kernel_size, padding = padding, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim = 1, keepdim = True)

        out = self.conv1(max_out)
        out = self.sigmoid(out)

        return out

class BAM(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(BAM, self).__init__()

        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca_out = self.ca(x)
        sa_out = self.sa(x)

        out = ca_out.mul(sa_out)*x

        return out
    