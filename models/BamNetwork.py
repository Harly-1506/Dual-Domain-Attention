import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock, Bottleneck, ResNet, resnet18

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}

from .BamRcan import *

class BamNet(ResNet):
    def __init__(self):
        super(BamNet, self).__init__(
           block =  BasicBlock, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 1000
           )
        

        state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)

        nn.init.kaiming_normal_(self.fc.weight)

        self.Myblock1 = BamNetwork(64,3,6,2) # 2-4
        self.Myblock2 = BamNetwork(128,3,6,2) # 3 -6
        self.Myblock3 = BamNetwork(256,3,6,2) # 4- 8
        self.Myblock4 = BamNetworklast(512,3,6) # 4 - 8
    

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)  # 56 :8
        m = self.Myblock1(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer2(x)  # 28 : 4 = 7
        m = self.Myblock2(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer3(x)  # 14 : 2 = 7
        m = self.Myblock3(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer4(x)  # 7
        m = self.Myblock4(x)
        x = x * (1 + m)
        # x = x * m

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def BamModel_dropout1(in_channels=3, num_classes=7):
    model = BamNet()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 7)
        # nn.Linear(512, num_classes)
    )
    return model