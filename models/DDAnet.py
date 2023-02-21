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

        # self.fc = nn.Linear(2048, 7)
        self.fc = nn.Linear(512, 7)

#         nn.init.kaiming_normal_(self.fc.weight)
#         nn.init.constant_(self.fc.bias, 0)

        self.DDA1 = DDABlock(64,512,1) # 2-4
        self.DDA2 = DDABlock(128,512,2) # 33-6
        self.DDA3 = DDABlock(256,512,3) # 4- 8
        self.DDA4 = DDABlock(512,512,4) # 4 - 8
        #resnet50
        # self.Myblock1 = Binet1(256,128) # 2-4
        # self.Myblock2 = Binet2(512,128) # 33-6
        # self.Myblock3 = Binet3(1024,128) # 4- 8
        # self.Myblock4 = Binet4(2048,128)
    

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)
        # print(x.size())  # 56 :8
        m = self.DDA1(x)
        x = x * (1 + m)


        x = self.layer2(x)
        # print(x.size())   # 28 : 4 = 7
        m = self.DDA2(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer3(x)  # 14 : 2 = 7
        m = self.DDA3(x)
        x = x * (1 + m)

        x = self.layer4(x)
        # print(x.size())   # 7
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
        # nn.Linear(512, num_classes)
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
        # self.fc = nn.Linear(512, 7)

        # nn.init.kaiming_normal_(self.fc.weight)


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
        # state_dict = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/resnet50_vggface2.pth")
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(2048, 7)
        # self.fc = nn.Linear(512, 7)

        # nn.init.kaiming_normal_(self.fc.weight)

        #570
        self.DDA1 = DDABlock(256, 612, 1)
        self.DDA2 = DDABlock(512, 612, 2) # 33-6
        self.DDA3 = DDABlock(1024, 612, 3) # 4- 8
        self.DDA4 = DDABlock(2048, 612, 4) # 4 - 8

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
        # print(x.size()) 
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


def DDAnet50_vggface_dropout2(in_channels=3, num_classes=7):
    model = DDAnet50_vggface()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(2048, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

class DDAnet34_MFC(ResNet):
    def __init__(self):
        super(DDAnet34_MFC, self).__init__(
           block =  BasicBlock, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 1000
           )
        

        state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        self.load_state_dict(state_dict)

      
#         nn.init.kaiming_normal_(self.fc.weight)
#         nn.init.constant_(self.fc.bias, 0)

        self.DDA1 = DDABlock(64,512,1) # 2-4
        self.DDA2 = DDABlock(128,512,2) # 33-6
        self.DDA3 = DDABlock(256,512,3) # 4- 8
        self.DDA4 = DDABlock(512,512,4) # 4 - 8
        #resnet50
        # self.Myblock1 = Binet1(256,128) # 2-4
        # self.Myblock2 = Binet2(512,128) # 33-6
        # self.Myblock3 = Binet3(1024,128) # 4- 8
        # self.Myblock4 = Binet4(2048,128)
    
        self.fc4 = nn.Linear(512, 512)
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(256,256)
        self.fc_out = nn.Linear(960, 7)
        
    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)
        # print(x.size())  # 56 : 64 x 64
        m = self.DDA1(x)
        x = x * (1 + m)

        f = self.avgpool(x)
        f = torch.flatten(f, 1)
        fc1 = self.fc1(f)
        # x = x * m

        x = self.layer2(x)
        # print(x.size())   # 28 x 128 x 128
        m = self.DDA2(x)
        x = x * (1 + m) 
        # x = x * m
        f = self.avgpool(x)
        f = torch.flatten(f, 1)
        fc2 = self.fc2(f)

        x = self.layer3(x)  # 14 x 256 x 256
        m = self.DDA3(x) 
        # print(x.size()) 
        x = x * (1 + m)
        # x = x * m
        f = self.avgpool(x)
        f = torch.flatten(f, 1)
        fc3 = self.fc3(f)

        x = self.layer4(x) 
        # print(x.size())   # 7 x 512
        m = self.DDA4(x)
        x = x * (1 + m)
        # x = x * m
        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        fc4 = self.fc4(f)
        
        fc_all = torch.cat((fc1,fc2,fc3,fc4), dim = 1)
        out = self.fc_out(fc_all)
#         print("out", out.size())
        return out

def DDAnet34_MFC_dropout1(in_channels=3, num_classes=7):
    model = DDAnet34_MFC()
    model.fc_out = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(960, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

class DDAnet50_vggface_MFC(ResNet):
    def __init__(self):
        super(DDAnet50_vggface_MFC, self).__init__(
           block =  Bottleneck, layers = [3, 4, 6, 3], in_channels = 3,num_classes = 8631
           )
        

        state_dict = load_state_dict_from_url(model_urls['vggface2'], progress=True)
        # state_dict = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/resnet50_vggface2.pth")
        self.load_state_dict(state_dict)

        self.fc1 = nn.Linear(256,128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512*14*14,256)
        self.fc3 = nn.Linear(1024*7*7,256)
        self.fc4 = nn.Linear(2048*4*4, 256)
        self.fc = nn.Linear(2048, 256)
        self.fc_out = nn.Linear(256*4, 7)

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
#         nn.init.kaiming_normal_(self.fc_out.weight)


        self.DDA1 = DDABlock(256,612,1)
        self.DDA2 = DDABlock(512,612,2) # 33-6
        self.DDA3 = DDABlock(1024,612,3) # 4- 8
        self.DDA4 = DDABlock(2048,612,4) # 4 - 8


    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)
        # print(x.size())  # 56 : 64 x 64
        m = self.DDA1(x)
        x = x * (1 + m)

#         f = self.maxpool(x)
#         f = torch.flatten(f, 1)
#         fc1 = self.fc1(f)
    

        x = self.layer2(x)
        # print(x.size())   # 28 x 128 x 128
        m = self.DDA2(x)
        x = x * (1 + m) 

        f = self.maxpool(x)
        f = torch.flatten(f, 1)
        fc2 = self.fc2(f)
        fc2 = self.relu_fc(fc2)

        x = self.layer3(x)  # 14 x 256 x 256
        m = self.DDA3(x) 
        # print(x.size()) 
        x = x * (1 + m)
        # x = x * m
        f = self.maxpool(x)
        f = torch.flatten(f, 1)
        fc3 = self.fc3(f)
        fc3 = self.relu_fc(fc3)
        
        x = self.layer4(x) 
        # print(x.size())   # 7 x 512
        m = self.DDA4(x)
        x = x * (1 + m)
        # x = x * m
        
        f = self.maxpool(x)
        f = torch.flatten(f, 1)
        fc4 = self.fc4(f)
        fc4 = self.relu_fc(fc4)
        
        f = self.avgpool(x)
        f = torch.flatten(f, 1)
        fc = self.fc(f)
        fc = self.relu_fc(fc)
        
        fc_all = torch.cat((fc2, fc3, fc4, fc), dim = 1)
        out = self.fc_out(fc_all)
#         print("out", out.size())
        return out


def DDAnet50_vggface_MFC_dropout1(in_channels=3, num_classes=7):
    model = DDAnet50_vggface_MFC()
    model.fc_out = nn.Sequential(
#         nn.Dropout(0.4),
        nn.Linear(256*4, num_classes)
        # nn.Linear(512, num_classes)
    )
    return model

