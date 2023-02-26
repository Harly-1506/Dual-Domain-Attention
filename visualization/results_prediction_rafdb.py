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
from torchvision import transforms

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.DDAnet import *
from utils.metrics.metrics import accuracy, make_batch

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class_names = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]

config_path = "/content/Research-Emotion/configs/config_rafdb.json"

configs = json.load(open(config_path))
test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 1) 
model = DDAnet50_vggface_dropout2()
# model = DDAnet50_dropout1()
# model = DDA_VGGnet_dropout1()
state = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/Rafdb_trainer_DDANet50_best_2023Jan31_05.49")
      
model.load_state_dict(state["net"])


def main():
    model.cuda()
    model.eval()
    total = 0
    test_set = RafDataSet("test", configs, ttau = True, len_tta = 64) 
    hold_test_set = RafDataSet("test", configs, ttau = False, len_tta = 0) 

    with torch.no_grad():
        for idx in tqdm.tqdm(range(100), total=100, leave=False):
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
                image = (image * torch.tensor(normalizer.std).view(3, 1, 1)) + torch.tensor(normalizer.mean).view(3, 1, 1)
                image = image.permute(1, 2, 0)
                image = np.uint8(image*255)
                plt.imsave(
                    "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/analyst_result/prediction_raf_wrong/{}->{}_{}.png".format(
                        class_names[target], class_names[outputs], idx
                    ),
                    image,
                )
            else:
                image, target = hold_test_set[idx]
                image = (image * torch.tensor(normalizer.std).view(3, 1, 1)) + torch.tensor(normalizer.mean).view(3, 1, 1)
                image = image.permute(1, 2, 0)
                image = np.uint8(image*255)
                plt.imsave(
                    "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/analyst_result/prediction_raf_correct/{}->{}_{}.png".format(
                        class_names[target], class_names[outputs], idx
                    ),
                    image,
                )
