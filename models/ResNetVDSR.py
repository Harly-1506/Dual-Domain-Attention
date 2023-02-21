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

from .srblock import VDSRblock

class VDResNet(ResNet):
    def __init__(self):
        super(VDResNet, self).__init__(
           block =  BasicBlock, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 1000
           )
        

        state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)

        self.vdblock1 = VDSRblock(64,64)
        self.vdblock2 = VDSRblock(128,128)
        self.vdblock3 = VDSRblock(256,256)
        self.vdblock4 = VDSRblock(512,512)

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)  # 56
        m = self.vdblock1(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer2(x)  # 28
        m = self.vdblock2(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer3(x)  # 14
        m = self.vdblock3(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer4(x)  # 7
        m = self.vdblock4(x)
        x = x * (1 + m)
        # x = x * m

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resnetvdsr_dropout1(in_channels=3, num_classes=7):
    model = VDResNet()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 7)
        # nn.Linear(512, num_classes)
    )
    return model