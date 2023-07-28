import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

seed = 113
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# from utils.datasets.fer2013_ds import FERDataset
# from utils.datasets.rafdb_ds import RafDataSet
# from models.vgg16_cbam import  VGG19_CBAM
# from models.resnet_cbam import ResidualNet , cbam_resnet50
from models.vggnet import vgg16_bn, vgg19, vgg19_bn, vgg16
from models.resnet import resnet50, resnet50_v2, resnet34
from models.vggnet_cbam import vgg16_cbam, vgg19_cbam
from models.vggnet_cbam_pre import vgg16_cbam_pre, vgg16_bn_cbam_pre, vgg19_bn_cbam_pre, MultiFCVGGnetCBam
from models.test_cbam import TestModel
from models.resmasking import *
from models.BamNetwork import *
from models.New_model import *
from models.DDAnet import *
from models.ResNetVDSR import resnetvdsr_dropout1

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt

image_path = "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/analyst_result/cam_rafdb/train_09051_aligned.jpg"


model =  DDAnet50_dropout1()

state = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/Rafdb_trainer_DDAnet50_2023Feb06_12.06")

model.load_state_dict(state["net"])

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def convert_tensor(tensor):
  # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    tensor = (tensor * torch.tensor(normalizer.std).view(3, 1, 1)) + torch.tensor(normalizer.mean).view(3, 1, 1)
    tensor = tensor.permute(1, 2, 0)
    tensor = np.uint8(tensor*255)
    # cv.imwrite(path, img)
    return tensor
def getGradCam(model, image_path, path):
    target_layers1 = [model.layer3]
    target_layers2 = [model.layer4]

    transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),

            ]
            )

    image = cv2.imread(image_path)
    print(image.shape)
    image = image[:, :, ::-1]
    print(image.shape)
    image1 = cv2.resize(image, (224, 224))

    tensor = transform(image)
 
    input_tensor = torch.unsqueeze(tensor, 0)

    # Construct the CAM object once, and then re-use it on many images:
    cam1 = GradCAMPlusPlus(model=model, target_layers=target_layers1, use_cuda=True)
    cam2 = GradCAMPlusPlus(model=model, target_layers=target_layers2, use_cuda=True)
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    targets = [ClassifierOutputTarget(3)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam1 = cam1(input_tensor=input_tensor, targets=None,eigen_smooth=False)
    grayscale_cam2 = cam2(input_tensor=input_tensor, targets=None,eigen_smooth=False)
    grayscale_cam1 = grayscale_cam1[0, :]
    grayscale_cam2 = grayscale_cam2[0, :]
    image = np.float32(image1) / 255
    visualization1 = show_cam_on_image(image, grayscale_cam1, use_rgb=True)
    visualization2 = show_cam_on_image(image, grayscale_cam2, use_rgb=True)
    
    plt.imsave(
        "/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/analyst_result/Thesis compare gradcam/Resnet50/Res50_{}.png".format(path),
        np.concatenate((convert_tensor(tensor), visualization1,visualization2), axis=1),
    )
    plt.imshow(np.concatenate((convert_tensor(tensor), visualization1,visualization2), axis=1))



raf_db_path = os.listdir("/content/dataset/aligned")
model = resnet50(out_classes = 7)
state = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/Rafdb_trainer_Resnet50_2023Feb06_11.01")
# model =  DDAnet50_dropout1()

# state = torch.load("/content/drive/MyDrive/WorkSpace/AI_Research/EmotionTorch/checkpoints/Rafdb_trainer_DDAnet50_2023Feb06_12.06")
model.load_state_dict(state["net"])
for i in range(len(raf_db_path)):
    getGradCam(model,"/content/dataset/aligned/{}".format(raf_db_path[i]), i )
    if i == 50:
        break
    
