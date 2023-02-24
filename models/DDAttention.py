import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LayersResnet18, MultiLevelAttention, ConvBlock, DualDomainFusion, up_pooling, Upsampler


class DDA1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_channels, 512)
        self.conv2 = ConvBlock(512, 512)
        self.conv3 = ConvBlock(512, 512) 

     
        self.layers = LayersResnet18(64, in_channels)
        self.arm1 = MultiLevelAttention(256, 256)
        self.arm2 = MultiLevelAttention(512, 512)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.ffm = DualDomainFusion(in_channels=256+512+512, out_channels=in_channels)
 
        self.up2 = Upsampler(scale = 3, in_channels = in_channels, out_channels = out_channels)
        self.outconv = nn.Conv2d(in_channels, in_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace = True)


                
                
        nn.init.kaiming_normal_(self.outconv.weight, mode="fan_out", nonlinearity="sigmoid")
        if self.outconv.bias is not None:
          nn.init.constant_(self.outconv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)



    def forward(self, x):

        pool2, pool3 = self.layers(x) 
        avg_pool = self.avg_pool(pool3)

        out_sp = self.conv1(x)
        out_sp = self.conv2(out_sp)
        out_sp = self.conv3(out_sp) 

        arm1 = self.arm1(pool2)
        arm2 = self.arm2(pool3)
        arm2 = torch.mul(arm2 , avg_pool) 

        
        arm1 = self.pool(arm1)
        out = torch.cat((arm1, arm2), dim = 1) 

        out = self.ffm(out_sp, out)


        out = self.up2(out)

        out = self.outconv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = torch.softmax(out, dim = 1)
        return out        

class DDA2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_channels, 512)
        self.conv2 = ConvBlock(512, 512)
        

        self.layers = LayersResnet18(128, in_channels)
        self.arm1 = MultiLevelAttention(256, 256)
        self.arm2 = MultiLevelAttention(512, 512)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.ffm = DualDomainFusion(in_channels=256 +512 + 512, out_channels=in_channels)
        self.up_arm2 =  Upsampler(scale = 1, in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)
        self.up2 = Upsampler(scale = 2, in_channels = in_channels, out_channels = out_channels)
        
        self.outconv = nn.Conv2d(in_channels, in_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace = True)
    
                
        nn.init.kaiming_normal_(self.outconv.weight, mode="fan_out", nonlinearity="relu")
        if self.outconv.bias is not None:
          nn.init.constant_(self.outconv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, x):

        pool1, pool2 = self.layers(x)
        avg_pool = self.avg_pool(pool2)

        out_sp = self.conv1(x)
        out_sp = self.conv2(out_sp)
 
        arm1 = self.arm1(pool1)
        arm2 = self.arm2(pool2)
        arm2 = torch.mul(arm2 , avg_pool) 


        arm2 = self.up_arm2(arm2)
    
        out = torch.cat((arm1, arm2), dim = 1) 

        out = self.ffm(out_sp, out) 

        out = self.up2(out)

        out = self.outconv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = torch.softmax(out, dim = 1)
        return out       

class DDA3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_channels, 512)
        self.layers = LayersResnet18(256, in_channels)
        
        self.arm1 = MultiLevelAttention(256, 256)
        self.arm2 = MultiLevelAttention(512, 512)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.ffm = DualDomainFusion(in_channels=256+ 512+ 512, out_channels=in_channels)
        self.up_arm2 =  Upsampler(scale = 1, in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)
        self.up2 = Upsampler(scale = 1, in_channels = in_channels, out_channels = out_channels)
        self.outconv = nn.Conv2d(in_channels, in_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace = True)
        
  
        nn.init.kaiming_normal_(self.outconv.weight, mode="fan_out", nonlinearity="relu")
        if self.outconv.bias is not None:
          nn.init.constant_(self.outconv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, x):

        pool1, pool2 = self.layers(x)
        avg_pool = self.avg_pool(pool2)

        out_sp = self.conv1(x)
 
        arm1 = self.arm1(pool1)
        arm2 = self.arm2(pool2)
        arm2 = torch.mul(arm2 , avg_pool) 

        sizesp = out_sp.size()[-2:]

        arm2 = self.up_arm2(arm2)
    
        out = torch.cat((arm1, arm2), dim = 1) 

        out = self.ffm(out_sp, out) 


        out = self.up2(out)
    
        out = self.outconv(out)
        out = self.bn(out)
        out = self.relu(out)

        out = torch.softmax(out, dim = 1)
        return out     

class DDA4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = LayersResnet18(512, in_channels)
        self.arm1 = MultiLevelAttention(512, 512)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.ffm = DualDomainFusion(in_channels=in_channels +512, out_channels=in_channels)
        self.up1 =  Upsampler(scale = 1, in_channels = 512, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)
        self.outconv = nn.Conv2d(in_channels, in_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace = True)
    
        nn.init.kaiming_normal_(self.outconv.weight, mode="fan_out", nonlinearity="relu")
        if self.outconv.bias is not None:
          nn.init.constant_(self.outconv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        pool = self.layers(x)
        avg_pool = self.avg_pool(pool)
        arm1 = self.arm1(pool)
        arm1 = torch.mul(arm1 , avg_pool) 
        arm1 = self.up1(arm1)
        out = self.ffm(x, arm1) 

        out = self.outconv(out)
        out = self.bn(out)
        out = self.relu(out)


        out = torch.softmax(out, dim = 1)
        return out 
    
def DDABlock(in_channels, out_channels, block):
    if block == 1:
        return DDA1(in_channels, out_channels)
    elif block == 2:
        return DDA2(in_channels, out_channels)
    elif block == 3:
        return DDA3(in_channels, out_channels)
    elif block == 4:
        return DDA4(in_channels, out_channels)
    else:
        print("error block")
