import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torchvision
import torch
from utils.augs.augmenters import seg_fer, seg_fertest1, seg_fertest2




EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FERDataset(Dataset):
  def __init__(self, data_type, configs, ttau = False, len_tta = 48):
    self.data_type = data_type
    self.configs = configs
    self.ttau = ttau
    self.len_tta = len_tta

    self.shape = (configs["image_size"], configs["image_size"])

    self.data = pd.read_csv(os.path.join(configs["data_path"]))

    if self.data_type == "train":
      self.data = self.data[self.data['Usage']=="Training"]

    if self.data_type == "val":
      self.data = self.data[self.data['Usage']=="PublicTest"]

    if self.data_type == "test":
      self.data = self.data[self.data['Usage']=="PrivateTest"]

    self.pixels = self.data["pixels"].tolist()

    self.emotions = pd.get_dummies(self.data["emotion"])

    self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    print(len(self.pixels))

  def is_ttau(self):
    return self.ttau == True

  def __len__(self):
    return len(self.pixels)
    
  def __getitem__(self, idx):

    pixel = self.pixels[idx]
    pixel = list(map(int, pixel.split(" ")))
    image = np.asarray(pixel).reshape(48, 48)
    image = image.astype(np.uint8)

    image = cv2.resize(image, self.shape)
    image = np.dstack([image] * self.configs["n_channels"])

    if self.data_type == "train":
      image = seg_fer(image = image)
        
    if self.data_type == "test" and self.ttau == True:
      images1 = [seg_fertest1(image=image) for i in range(self.len_tta)]
      images2 = [seg_fertest2(image=image) for i in range(self.len_tta)]

      images = images1 + images2
      # images = [image for i in range(self._tta_size)]
      images = list(map(self.transform, images))
      label = self.emotions.iloc[idx].idxmax()
      return images, label

    image = self.transform(image)
    label = self.emotions.iloc[idx].idxmax()
    
    return image, label


# class FERDataset(Dataset):
#   def __init__(self, data_type, configs, ttau = False, len_tta = 48):
#     self.data_type = data_type
#     self.configs = configs
#     self.ttau = ttau
#     self.len_tta = len_tta

#     self.shape = (configs["image_size"], configs["image_size"])

#     self.data = pd.read_csv(os.path.join(configs["data_path"]))

#     if self.data_type == "train":
#       self.data = self.data[self.data['Usage']=="Training"]

#     if self.data_type == "val":
#       self.data = self.data[self.data['Usage']=="PublicTest"]

#     if self.data_type == "test":
#       self.data = self.data[self.data['Usage']=="PrivateTest"]

#     self.pixels = self.data["pixels"].tolist()

#     self.emotions = pd.get_dummies(self.data["emotion"])

#     self.transform = transforms.Compose(
#         [
#             transforms.ToPILImage(),
#             transforms.ToTensor(),
#         ]
#     )
#     print(len(self.pixels))

#   def is_ttau(self):
#     return self.ttau == True

#   def __len__(self):
#     return len(self.pixels)
    
#   def __getitem__(self, idx):

#     pixel = self.pixels[idx]
#     pixel = list(map(int, pixel.split(" ")))
#     image = np.asarray(pixel).reshape(48, 48)
#     image = image.astype(np.uint8)

#     image = cv2.resize(image, self.shape)
#     image = np.dstack([image] * self.configs["n_channels"])

#     if self.data_type == "train":
#       image = transform_al(image = image)["image"]

#     if self.data_type == "test" and self.ttau == True:
#       images = [transform_al(image = image)["image"] for i in range(self.len_tta)]
#       # images = [image for i in range(self._tta_size)]
#       images = list(map(self.transform, images))
#       label = self.emotions.iloc[idx].idxmax()
#       return images, label

#     image = self.transform(image)
#     label = self.emotions.iloc[idx].idxmax()
    
#     return image, label
