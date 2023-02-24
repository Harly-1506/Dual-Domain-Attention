import torch
import torch.nn as nn


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .resnet import BasicBlock, Bottleneck, ResNet, resnet18
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "vggface2": "https://onedrive.live.com/download?cid=D07B627FBE5CFFFA&resid=D07B627FBE5CFFFA%21587&authkey=APXT_JMvytW7cgk",
}

from .DDAttention import DDABlock


class DDANet34(ResNet):
    def __init__(self):
        super(DDANet34, self).__init__(
           block =  BasicBlock, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 1000
           )
        

        state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)



        self.DDA1 = DDABlock(64,512,1) 
        self.DDA2 = DDABlock(128,512,2) 
        self.DDA3 = DDABlock(256,512,3) 
        self.DDA4 = DDABlock(512,512,4) 

    

    def forward(self, x):  
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  

        x = self.layer1(x)
        m = self.DDA1(x)
        x = x * (1 + m)


        x = self.layer2(x)
        m = self.DDA2(x)
        x = x * (1 + m)


        x = self.layer3(x) 
        m = self.DDA3(x)
        x = x * (1 + m)

        x = self.layer4(x)
        m = self.DDA4(x)
        x = x * (1 + m)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def DDAnet34_dropout1(in_channels=3, num_classes=7):
    model = DDANet34()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


class DDAnet50(ResNet):
    def __init__(self):
        super(DDAnet50, self).__init__(
           block =  Bottleneck, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 1000
           )
        

        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(2048, 7)



        self.DDA1 = DDABlock(256,612,1)
        self.DDA2 = DDABlock(512,612,2) 
        self.DDA3 = DDABlock(1024,612,3) 
        self.DDA4 = DDABlock(2048,612,4)

    

    def forward(self, x):  
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  

        x = self.layer1(x)
        m = self.DDA1(x)
        x = x * (1 + m)
        

        x = self.layer2(x)
        m = self.DDA2(x)
        x = x * (1 + m)


        x = self.layer3(x)  
        m = self.DDA3(x)
        x = x * (1 + m)


        x = self.layer4(x)
        m = self.DDA4(x)
        x = x * (1 + m)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def DDAnet50_dropout1(in_channels=3, num_classes=7):
    model = DDAnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model


class DDAnet50_vggface(ResNet):
    def __init__(self):
        super(DDAnet50_vggface, self).__init__(
           block =  Bottleneck, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 8631
           )
        

        state_dict = load_state_dict_from_url(model_urls['vggface2'], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(2048, 7)



        self.DDA1 = DDABlock(256,612, 1)
        self.DDA2 = DDABlock(512,612, 2) 
        self.DDA3 = DDABlock(1024,612, 3) 
        self.DDA4 = DDABlock(2048,612, 4) 

    def forward(self, x):  
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  

        x = self.layer1(x)
        m = self.DDA1(x)
        x = x * (1 + m)
  

        x = self.layer2(x)
        m = self.DDA2(x)
        x = x * (1 + m)


        x = self.layer3(x)  
        m = self.DDA3(x)
        x = x * (1 + m)
   

        x = self.layer4(x)
        m = self.DDA4(x)
        x = x * (1 + m)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def DDAnet50_vggface_dropout2(in_channels=3, num_classes=7):
    model = DDAnet50_vggface()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

