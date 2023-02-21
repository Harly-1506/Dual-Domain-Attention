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

# flat = [64*224*224, 128*112*112, 256*56*56, 512*28*28, 512*14*14]

class Cbam(nn.Module):
    def __init__(self, in_channels):
        super(Cbam, self).__init__()

        self.Cbam = CBAM(in_channels)
        self.relu = nn.ReLU(inplace = True)

        for key in self.state_dict():
            # print(key)
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0   

    def forward(self, x):
        x = self.Cbam(x)
        # x = self.relu(x)
        return x

class ResCbam(nn.Module):
    def __init__(self, in_channels):
        super(ResCbam, self).__init__()

        self.Cbam = CBAM(in_channels)
        self.relu = nn.ReLU(inplace = True)
        
        for key in self.state_dict():
            # print(key)
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def forward(self, x):
        res = x
        x = self.Cbam(x)
        # x = x*(1+res)
        x += res
        x = self.relu(x)
        return x   



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

class VGGBN16CBam(BackBone):
    def __init__(self):
        super(VGGBN16CBam, self).__init__(make_vgg_layers(vgg16_config, batch_norm =  True))

        state_dict = load_state_dict_from_url(model_urls["vgg16_bn"], progress = True)
        self.load_state_dict(state_dict)

        k = 0
        for i in range(len(vgg16_config)):
            if vgg16_config[i] == "M":
                k +=1 
                self.layers.insert(k, Cbam(vgg16_config[i-1]))
                k += 1
            else:
                k += 3


        self.features = nn.Sequential(*self.layers)
        self.classifier = nn.Sequential(*self.fc_layers)

        # for key in self.state_dict():     
        #     print(key
        #     if key.slit('.')[-1]=="weight":
        #         if bn" in key:
        #            if "SpatialGate" in key:
        #                self.state_dict()[key][...] = 0
                

    def forward(self, x):
        
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x 
        

class VGG16CBam(BackBone):
    def __init__(self):
        super(VGG16CBam, self).__init__(make_vgg_layers(vgg16_config, batch_norm = False))

        state_dict = load_state_dict_from_url(model_urls["vgg16"], progress = True)
        self.load_state_dict(state_dict)

        cbam1 = CBAM(64)
        cbam2 = CBAM(128)
        cbam3 = CBAM(256)
        cbam4 = CBAM(512)
        cbam5 = CBAM(512)

        cbam_layer = [cbam1, cbam2, cbam3, cbam4, cbam5]
        
        c = 0
        k = 0

        cbam_layer = [cbam1, cbam2, cbam3, cbam4, cbam5]

        for i in range(len(vgg16_config)):
            if vgg16_config[i] == "M":
                self.layers.insert(k, cbam_layer[c])
                c += 1
            k += 2

        self.features = nn.Sequential(*self.layers)

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class VGGBN19CBam(BackBone):
    def __init__(self):
        super(VGGBN19CBam, self).__init__(make_vgg_layers(vgg19_config, batch_norm = True))

        state_dict = load_state_dict_from_url(model_urls["vgg19_bn"], progress = True)
        self.load_state_dict(state_dict)


        cbam1 = CBAM(64)
        cbam2 = CBAM(128)
        cbam3 = CBAM(256)
        cbam4 = CBAM(512)
        cbam5 = CBAM(512)

        cbam_layer = [cbam1, cbam2, cbam3, cbam4, cbam5]
        c = 0
        k = 0

        for i in range(len(vgg19_config)):
            if vgg19_config[i] == "M":
                self.layers.insert(k, cbam_layer[c])
                c += 1
                k += 2
            else:
                k += 3

        self.features = nn.Sequential(*self.layers)

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class VGG19CBam(BackBone):
    def __init__(self):
        super(VGG19CBam, self).__init__(make_vgg_layers(vgg19_config, batch_norm = False))

        state_dict = load_state_dict_from_url(model_urls["vgg19"], progress = True)
        self.load_state_dict(state_dict)


        cbam1 = CBAM(64)
        cbam2 = CBAM(128)
        cbam3 = CBAM(256)
        cbam4 = CBAM(512)
        cbam5 = CBAM(512)
        c = 0
        k = 0

        cbam_layer = [cbam1, cbam2, cbam3, cbam4, cbam5]

        for i in range(len(vgg19_config)):
            if vgg19_config[i] == "M":
                self.layers.insert(k, cbam_layer[c])
                c += 1
            k += 2

        self.features = nn.Sequential(*self.layers)

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class MultiFCVGGnetCBam(nn.Module):
    def __init__(self, model):
        super(MultiFCVGGnetCBam, self).__init__()

        
        self.feature1 = torch.nn.Sequential(*list(model.features.children())[:8])
        self.feature2 = torch.nn.Sequential(*list(model.features.children())[8:16])
        self.feature3 = torch.nn.Sequential(*list(model.features.children())[16:27])
        self.feature4 = torch.nn.Sequential(*list(model.features.children())[27:38])
        self.feature5 = torch.nn.Sequential(*list(model.features.children())[38:])
        
        self.fc1 = nn.Linear(64*112*112, 256)
        self.relu1 = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(128*56*56, 256)
        self.relu2 = nn.ReLU(inplace = True)
        self.fc3 = nn.Linear(256*28*28, 256)
        self.relu3 = nn.ReLU(inplace = True)
        self.fc4 = nn.Linear(512*14*14, 256)
        self.relu4 = nn.ReLU(inplace = True)
        self.fc5 = nn.Linear(512*7*7, 256)
        self.relu5 = nn.ReLU(inplace = True)

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.fc_layers = [
            nn.Linear(512 * 7 * 7, 1208),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(1208, 1208),
            nn.ReLU(inplace = True),

        ]
        self.classifier = nn.Sequential(*self.fc_layers)
        
        self.out = nn.Linear(2488,7)    
        nn.init.kaiming_normal_(self.out.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature1(x)
        m1 = torch.flatten(x,1)
        fc1 = self.fc1(m1)
        fc1 = self.relu1(fc1)

        x = self.feature2(x)
        m2 = torch.flatten(x,1)
        fc2 = self.fc2(m2)
        fc2 = self.relu2(fc2)
        # concat1 = torch.cat([fc1,fc2])

        x = self.feature3(x)
        m3 = torch.flatten(x,1)
        fc3 = self.fc3(m3)
        fc3 = self.relu3(fc3)
        # concat2 = torch.cat([concat1,fc3], dim = 1)

        x = self.feature4(x)
        m4 = torch.flatten(x,1)
        fc4 = self.fc4(m4)
        fc4 = self.relu4(fc4)
        # concat3 = torch.cat([concat2, fc4], dim =1)

        x = self.feature5(x)
        m5 = torch.flatten(x,1)
        fc5 = self.fc5(m5)
        fc5 = self.relu5(fc5)
        concat = torch.cat([fc1,fc2,fc3,fc4, fc5], dim =1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        out = torch.cat([concat, x], dim = 1)
        out = self.out(out)

        return out



def load_vgg(url_net, config, batch_norm, pretrained, progress):

    model = BackBone(make_vgg_layers(config, batch_norm = batch_norm))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[url_net], progress = progress)
        model.load_state_dict(state_dict)

    return model


def vgg16_bn_test(pretrained = True, batch_norm =True):
    model = load_vgg("vgg16_bn", vgg16_config, batch_norm=True, pretrained= pretrained, progress= True)

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 7),
    )
    return model



def vgg16_bn_cbam_pre(num_classes):
    model = VGGBN16CBam()

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
    return model


def vgg16_cbam_pre(num_classes):
    model = VGG16CBam()

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
    return model
    

def vgg19_bn_cbam_pre(num_classes):
    model = VGGBN19CBam()

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
    return model


def vgg19_cbam_pre(num_classes):
    model = VGG19CBam()

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
    return model

