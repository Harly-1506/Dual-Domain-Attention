import torch
import torch.nn as nn
from .DDAttention import DDABlock

Wide_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
ResNext50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)


class DDAWide_net50(nn.Module):
    def __init__(self, model):
        super(DDAWide_net50, self).__init__()
        
        self.model = model
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool

        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
        self.avgpool = self.model.avgpool
 

        self.fc = nn.Linear(2048, 7)



        self.DDA1 = DDABlock(256,612,1)
        self.DDA2 = DDABlock(512,612,2) # 33-6
        self.DDA3 = DDABlock(1024,612,3) # 4- 8
        self.DDA4 = DDABlock(2048,612,4) # 4 - 8

#         self.Myblock1 = Binet1(256,256) # 2-4
#         self.Myblock2 = Binet2(512,256) # 33-6
#         self.Myblock3 = Binet3(1024,256) # 4- 8
#         self.Myblock4 = Binet4(2048,256)
    

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)
        # print(x.size())  # 56 :8
        m = self.DDA1(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer2(x)
        # print(x.size())   # 28 : 4 = 7
        m = self.DDA2(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer3(x)  # 14 : 2 = 7
        m = self.DDA3(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer4(x)
        # print(x.size())   # 7
        m = self.DDA4(x)
        x = x * (1 + m)
        # x = x * m

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x
        

def Wide_resnet50_dropout(in_channels = 3, num_classes =7):
    model = Wide_resnet50
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

def DDAWidenet50_dropout1(in_channels=3, num_classes=7):
    model = DDAWide_net50(Wide_resnet50)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

def Resnext50_dropout(in_channels = 3, num_classes =7):
    model = ResNext50
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

def DDAResnext50_dropout1(in_channels=3, num_classes=7):
    model = DDAWide_net50(ResNext50)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model
