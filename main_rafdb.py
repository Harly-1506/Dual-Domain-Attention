import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.datasets.fer2013_ds import FERDataset
from utils.datasets.rafdb_ds import RafDataSet
# from models.vgg16_cbam import  VGG19_CBAM
from models.resnet_cbam import ResidualNet , cbam_resnet50
from models.vggnet import vgg16_bn, vgg19, vgg19_bn, vgg16
from models.resnet import resnet50, resnet50_v2, resnet34
from models.vggnet_cbam import vgg16_cbam, vgg19_cbam
from models.DDAnet_vgg import *
from models.test_cbam import TestModel
from models.resmasking import *
from models.BamNetwork import *
from models.New_model import *
from models.DDAnet import *

from utils.visualize.show_img import show_image_dataset
from trainer.fer2013_trainer import FER2013_Trainer
from trainer.rafdb_trainer import RAFDB_Trainer
from trainer.rafdb_trainer_KFold import RAFDB_Trainer_KFold

print(torch.__version__)

config_path = "configs/config_rafdb.json"

configs = json.load(open(config_path))

train_loader = RafDataSet( "train", configs)
# val_loader = RafDataSet("val", configs)
test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 
test_loader = RafDataSet("test", configs, ttau = False, len_tta = 48) 

# show_image_dataset(train_ds)

# model = resnet50_cbam()
# if torch.cuda.is_available():
#     model.cuda()

# n_inputs = model.classifier[6].in_features  
# model.classifier[6] = nn.Sequential(
#             nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(256, 7))

# import torchvision
# model = vgg16_bn()
model = resnet34()
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

# model = resnet34(num_classes = 7)
trainer = RAFDB_Trainer(model, train_loader, test_loader, test_loader, test_loader_ttau, configs , wb = False)

#760c20e6aae9012f4e41e4feffa73c63d4a10fc3

# trainer.acc_on_test()
# trainer.acc_on_test_ttau()
# 
trainer.Train_model()
