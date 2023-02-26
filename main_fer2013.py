import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.datasets.fer2013_ds import FERDataset
from models.DDAnet import *
# from models.vgg16_cbam import  VGG19_CBAM
# from models.resnet_cbam import ResidualNet , cbam_resnet50
# from models.vggnet import vgg16_bn, vgg19, vgg19_bn, vgg16
# from models.resnet import resnet50
# from models.vggnet_cbam import vgg16_cbam, vgg19_cbam
# from models.vggnet_cbam_pre import vgg16_cbam_pre, vgg16_bn_cbam_pre, vgg19_bn_cbam_pre, MultiFCVGGnetCBam
# from models.test_cbam import TestModel
# from models.resmasking import *
# from models.BamNetwork import *
# from models.New_model import *
# from models.Binetwork import *
# from models.ResNetVDSR import resnetvdsr_dropout1

from utils.visualize.show_img import show_image_dataset
from trainer.fer2013_trainer import FER2013_Trainer

print(torch.__version__)

config_path = "/kaggle/working/Research-Emotion/configs/config_fer2013.json"

configs = json.load(open(config_path))

train_loader = FERDataset( "train", configs)
val_loader = FERDataset("val", configs)
test_loader_ttau = FERDataset("test", configs, ttau = True, len_tta = 10) 
test_loader = FERDataset("test", configs, ttau = False, len_tta = 48)

# show_image_dataset(train_ds)

# model = resnet50_cbam()
# if torch.cuda.is_available():
#     model.cuda()

# n_inputs = model.classifier[6].in_features  
# model.classifier[6] = nn.Sequential(
#             nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(256, 7))

# import torchvision
# model1 = vgg16_bn()
# model = resnet50(True, True, num_classes = 7)
# model1 = ResidualNet("ImageNet", 50, 7, "CBAM")
# model = cbam_resnet50(in_channels=3, num_classes= 7 )
# model = vgg19()
# model = vgg19_bn()
# model1 = vgg16_bn(pretrained = True, batch_norm = True)
# model = vgg19_cbam(num_classes = 7)
# model = vgg16_bn_cbam_pre(num_classes=7)
# pretrained_model = torchvision.models.vgg16_bn(pretrained=True)
# model = TestModel(pretrained_model)
# model = MultiFCVGGnetCBam(model1)
# model = resmasking_dropout1()
# model = resnetvdsr_dropout1()
# model = DDAnet34_MFC_dropout1()
# model = DDAnet50_vggface_dropout2()
model = DDAnet34_dropout1()


trainer = FER2013_Trainer(model, train_loader, val_loader, test_loader, test_loader_ttau, configs , wb = False)

# trainer.acc_on_test()
# trainer.acc_on_test_ttau()

trainer.Train_model()