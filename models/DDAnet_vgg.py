# dataiter = iter(train_ds)
# images, labels = dataiter.next()
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .DDAttention import DDABlock


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M']

def make_vgg_layers(vgg_config, batch_norm):

    layers = []
    in_channels = 3

    for c in vgg_config:
        assert c == "M" or isinstance(c, int)
        if c == "M":
            layers += [nn.MaxPool2d(kernel_size = 2)]
        
        else:
            conv2d =  (nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]

            in_channels = c

    return layers


class BackBone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.layers = layers

        self.features = nn.Sequential(*self.layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.fc_layers = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        ]
        self.classifier = nn.Sequential(*self.fc_layers)
        # self._initialize_weights()
        # print(self.features)
    def forward(self, x):
       
        x = self.features(x)
        # x = layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                
class DDA_VGGnet(nn.Module):
    def __init__(self, model):
        super(DDA_VGGnet, self).__init__()

        
        self.feature1 = torch.nn.Sequential(*list(model.features.children())[:7])
        self.feature2 = torch.nn.Sequential(*list(model.features.children())[7:14])
        self.feature3 = torch.nn.Sequential(*list(model.features.children())[14:24])
        self.feature4 = torch.nn.Sequential(*list(model.features.children())[24:34])
        self.feature5 = torch.nn.Sequential(*list(model.features.children())[34:])

        self.DDA1 = DDABlock(128,512,1) # 2-4
        self.DDA2 = DDABlock(256,512,2) # 33-6
        self.DDA3 = DDABlock(512,512,3) # 4- 8
        self.DDA4 = DDABlock(512,512,4)
   
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.fc_layers = [
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.4),
            nn.Linear(512, 7)

        ]
        self.classifier = nn.Sequential(*self.fc_layers)
           

    def forward(self, x):

        x = self.feature1(x)
#         print("size1", x.size())

        x = self.feature2(x)
#         print("size2", x.size())
        m = self.DDA1(x)
        x = x * (1 + m)

        x = self.feature3(x)
#         print("size3",x.size())
        m = self.DDA2(x)
        x = x * (1 + m)

        x = self.feature4(x)
#         print("size4",x.size())
        m = self.DDA3(x)
        x = x * (1 + m)

        x = self.feature5(x)
#         print("size5",x.size())
        m = self.DDA4(x)
        x = x * (1 + m)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x



def load_vgg(url_net, config, batch_norm, pretrained, progress):

    model = BackBone(make_vgg_layers(config, batch_norm = batch_norm))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[url_net], progress = progress)
        model.load_state_dict(state_dict)

    return model


def vgg16_bn(pretrained = True, batch_norm =True):

    model = load_vgg("vgg16_bn", vgg16_config, batch_norm=True, pretrained= pretrained, progress= True)

    return model

def DDA_VGGnet_dropout1():
    vgg16 = vgg16_bn()
    model = DDA_VGGnet(vgg16)

    return model


