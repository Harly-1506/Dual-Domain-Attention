import torch
import torch.nn as nn
import math
from models.resnet import BasicBlock, Bottleneck, ResNet

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


def up_pooling(in_channels, out_channels, kernel_size=3, stride=2, padding = 1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = padding, bias = False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class Upsampler(nn.Module):
    def __init__(self, scale, in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0):
        super(Upsampler, self).__init__()
        
        self.scale = scale
        self.in_channels = in_channels
        layers = []
        if self.scale > 1:
            for i in range(self.scale - 1):
                layers.append(up_pooling(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding))
                in_channels = out_channels
            
            self.uplayers = nn.Sequential(*layers)
            self.upout = up_pooling(out_channels, self.in_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        else:
            self.upout = up_pooling(in_channels, self.in_channels, kernel_size = kernel_size, stride = stride, padding = padding)
            
        for m in self.modules():
          if isinstance(m, nn.ConvTranspose2d):
             nn.init.kaiming_normal_(m.weight,  mode="fan_out", nonlinearity="relu")
          elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
             nn.init.constant_(m.weight, 1)
             nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        if self.scale > 1:
            out = self.uplayers(x)
            out = self.upout(out)
            return out
        else:
            return self.upout(x)
        
class LayersResnet18(ResNet):
    def __init__(self, numlayers, out_channels):
        super(LayersResnet18, self).__init__(
           block =  BasicBlock, layers = [2,2,2,2], in_channels = 3,num_classes = 1000
           )
        
        
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        self.load_state_dict(state_dict)
        self.num = numlayers
        self.conv = nn.Conv2d(out_channels, numlayers, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(numlayers)
        self.relu = nn.ReLU(inplace = True)
        if numlayers == 64:
            self.conv2 = nn.Conv2d(numlayers, 64, kernel_size = 3, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(64)
        else:
            self.conv2 = nn.Conv2d(numlayers, numlayers //2 , kernel_size = 3, padding = 1, bias = False)    
            self.bn2 = nn.BatchNorm2d(numlayers // 2)
        self.relu2 = nn.ReLU(inplace = True)
        
        
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):  # 224
        
        if self.num  == 64:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            out = self.layer1(x) 
            out1 = self.layer2(out) 
            out2 = self.layer3(out1)  
            out3 = self.layer4(out2)  
            return out2, out3  
        elif self.num  == 128:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            out1 = self.layer2(x)  
            out2 = self.layer3(out1) 
            out3 = self.layer4(out2) 
            return out2 ,out3
        elif self.num  == 256: 
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            out3 = self.layer3(x)  
            out4 = self.layer4(out3)
            return out3, out4
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            out4 = self.layer4(x)

            return out4

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels,eps=1e-5, affine=True) if bn else None
        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, bias = False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):

        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out



class MultiLevelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="sigmoid")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        
    def forward(self, x):

        out = self.avg_pool(x)
        assert self.in_channels == out.size(1), 'in_channels and out_channels should all be {}'.format(out.size(1))
        out = self.conv(out)
        out = self.bn(out)
        out = torch.softmax(out, dim = 1)

        return torch.mul(x, out)

class DualDomainFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=1)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        if self.conv1.bias is not None:
           nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
                

        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="sigmoid")
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        
    def forward(self, x_input_1, x_input_2):
        x = torch.cat((x_input_1, x_input_2), dim=1) 
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.conv_block(x) 
        
        x = self.avg_pool(feature)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.softmax(x, dim = 1)

        x = torch.mul(feature, x)
        
        return torch.add(feature, x)
