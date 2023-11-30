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


config_path = "configs/config_rafdb.json"

configs = json.load(open(config_path))

test_loader_ttau = RafDataSet("test", configs, ttau = True, len_tta = 10) 


model =  DDAnet50_vggface_dropout2()
state = torch.load("Rafdb_trainer_DDANet50_best_2023Jan31_05.49")
      
model.load_state_dict(state["net"])

def plot_confusion_matrix(model, testloader,title = "My model"):
    model.cuda()
    model.eval()

    correct = 0
    total = 0
    all_target = []
    all_output = []

    # test_set = fer2013("test", configs, tta=True, tta_size=8)
    # test_set = fer2013('test', configs, tta=False, tta_size=0)

    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(testloader)), total=len(testloader), leave=False):
            images, labels = testloader[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            preds = model(images).cpu()
            preds = F.softmax(preds, 1)

            # preds.shape [tta_size, 7]
            preds = torch.sum(preds, 0)
            preds = torch.argmax(preds, 0)
            preds = preds.item()
            labels = labels.item()
            total += 1
            correct += preds == labels

            all_target.append(labels)
            all_output.append(preds)

    
    cf_matrix = confusion_matrix(all_target, all_output)
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    class_names = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
    #0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

    # Create pandas dataframe
    dataframe = pd.DataFrame(cmn, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=True,cmap="Blues",fmt=".2f")
    
    plt.title(title), plt.tight_layout()
    
    plt.ylabel("True Class", fontsize=12), 
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()

    plt.savefig("DDAnet50_vggface_RAFDB_CM.pdf")
    plt.close()
if __name__ == '__main__':
  plot_confusion_matrix(model, test_loader_ttau)
