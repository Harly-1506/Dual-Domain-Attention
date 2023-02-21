import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import cv2
import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.metrics.metrics import accuracy, make_batch
from utils.datasets.fer2013_ds import FERDataset
# from models.vgg16_cbam import  VGG19_CBAM
from models.resnet_cbam import ResidualNet , cbam_resnet50
from models.vggnet import vgg16_bn, vgg19, vgg19_bn, vgg16
from models.resnet import resnet50
from models.vggnet_cbam import vgg16_cbam, vgg19_cbam
from models.vggnet_cbam_pre import vgg16_cbam_pre, vgg16_bn_cbam_pre, vgg19_bn_cbam_pre, MultiFCVGGnetCBam
from models.test_cbam import TestModel
from models.resmasking import *
from models.BamNetwork import *
from models.New_model import *
from models.Binetwork import *
from models.ResNetVDSR import resnetvdsr_dropout1

from utils.visualize.show_img import show_image_dataset
from trainer.fer2013_trainer import FER2013_Trainer

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

config_path = "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/code/configs/config_fer2013.json"
configs = json.load(open(config_path))
test_loader_ttau = FERDataset("test", configs, ttau = True, len_tta = 64) 

model =  Binet_dropout1()
state = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/Fer2013_trainers_Binet2_Resnet34_Bi18_adam_relu11_27_2022Nov27_04.57_74.171.57")
      
model.load_state_dict(state["net"])

def main():
    model.cuda()
    model.eval()
    total = 0
    test_set = FERDataset("test", configs, ttau = True, len_tta = 64) 
    hold_test_set = FERDataset("test", configs, ttau = False, len_tta = 0) 

    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, targets = test_set[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            outputs = model(images).cpu()
            outputs = F.softmax(outputs, 1)

            outputs = torch.sum(outputs, 0)
            outputs = torch.argmax(outputs, 0)
            outputs = outputs.item()
            targets = targets.item()
            total += 1
            if outputs != targets:
                image, target = hold_test_set[idx]
                image = image.permute(1, 2, 0).numpy() * 255
                image = image.astype(np.uint8)

                cv2.imwrite(
                    "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/analyst_result/prediction_fer_wrong/{}->{}_{}.png".format(
                        class_names[target], class_names[outputs], idx
                    ),
                    image,
                )
            else:
                image, target = hold_test_set[idx]
                image = image.permute(1, 2, 0).numpy() * 255
                image = image.astype(np.uint8)

                cv2.imwrite(
                    "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/analyst_result/prediction_fer_correct/{}->{}_{}.png".format(
                        class_names[target], class_names[outputs], idx
                    ),
                    image,
                )
