import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from .resnet import conv1x1, conv3x3, BasicBlock, Bottleneck

filters = []


class BasicConv(nn.Module):
    def __init__(self, in_channels):
        super(BasicConv, self).__init__()
        
        '''
        SRCNN for image super resolution
        '''

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, stride = 1, bias = False)
    def forward(self, x):
        # create reslock with srcnn to get more accuracy
        res = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = torch.add(out, res)

        return out

class VDSRblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        assert in_channels == out_channels
        super(VDSRblock, self).__init__()

        "Very Deep Super Block for super image resolution"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.residual_layer = self.make_layers(BasicConv, 9)
        
        self.input = nn.Conv2d(in_channels = self.in_channels, out_channels = 1 , kernel_size = 1, stride = 1, padding = 0)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = self.out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False )
        self.conv2 = nn.Conv2d(in_channels = self.out_channels, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.output = nn.Conv2d(in_channels = 1, out_channels = self.out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU(inplace = True)


        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2./n))

    
        

    def make_layers(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.in_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        res = x
        out = self.relu(self.input(x))
        out = self.conv1(out)
        out = self.residual_layer(out)
        out = self.conv2(out)

        out = self.output(out)

        out = torch.add(out, res)

        out = torch.softmax(out, dim = 1)

        return out