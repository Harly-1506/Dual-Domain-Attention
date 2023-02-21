import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from models.cbam import *


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

class TestModel(nn.Module):
    def __init__(self, model):
        super(TestModel, self).__init__()

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)


        self.features_1 = torch.nn.Sequential(*list(model.features.children())[0:7])
        self.features_2 = torch.nn.Sequential(*list(model.features.children())[7:14])
        self.features_3 = torch.nn.Sequential(*list(model.features.children())[14:24])
        self.features_4 = torch.nn.Sequential(*list(model.features.children())[24:34])
        self.features_5 = torch.nn.Sequential(*list(model.features.children())[34:])
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.fc_layers = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 7)
        ]
        self.classifier = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.features_1(x)
        x = self.cbam1(x)

        x = self.features_2(x)
        x = self.cbam2(x)

        x = self.features_3(x)
        x = self.cbam3(x)     

        x = self.features_4(x)
        x = self.cbam4(x)    

        x = self.features_5(x)
        x = self.cbam5(x)       

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

