import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torchvision
import torch
from utils.augs.augmenters import seg_fer, seg_fertest1, seg_fertest2


def majority_vote(list_labels):
  new_list_labels = np.argmax(list_labels, axis = 1)

  return new_list_labels

def get_labels(data_labels, data):
    emotions = np.array(data_labels[orig_class_names])
    
    y_mask = emotions.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('NF')
    emotions = emotions[mask]
    data = data[mask]
    # Convert to probabilities between 0 and 1
    emotions = emotions[:, :-1] * 0.1

    # Add contempt to neutral and remove it
#     emotions[:, 0] += emotions[:, 7]
    emotions[:, 0] += emotions[:, 8]
    emotions = emotions[:, :8]
    return emotions, data

orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt','unknown', 'NF']

class FERPlusDataset(Dataset):
  def __init__(self, data_type, configs, ttau = False, len_tta = 48):
    self.data_type = data_type
    self.configs = configs
    self.ttau = ttau
    self.len_tta = len_tta

    self.shape = (configs["image_size"], configs["image_size"])

    self.data = pd.read_csv(os.path.join(configs["data_path"]))
    self.data_labels = pd.read_csv(os.path.join(configs["data_labels"]))

    if self.data_type == "train":
      self.data = self.data[self.data['Usage']=="Training"]
      self.data_labels = self.data_labels[self.data_labels['Usage']=="Training"]
    if self.data_type == "val":
      self.data = self.data[self.data['Usage']=="PublicTest"]
      self.data_labels = self.data_labels[self.data_labels['Usage']=="PublicTest"]
    if self.data_type == "test":
      self.data = self.data[self.data['Usage']=="PrivateTest"]
      self.data_labels = self.data_labels[self.data_labels['Usage']=="PrivateTest"]

    
    self.emotions, self.data = get_labels(self.data_labels, self.data)
    self.pixels = self.data["pixels"].tolist()
    self.emotions = majority_vote(self.emotions)
#     print(len(self.emotions))
    # self.emotions = pd.get_dummies(self.data["emotion"])

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
      images2 = [seg_fertest2(image=image) for i in range(self.len_tta//2)]

      images = images1 + images2
      # images = [image for i in range(self._tta_size)]
      images = list(map(self.transform, images))
      label = self.emotions[idx]
      return images, label

    image = self.transform(image)
    label = self.emotions[idx]
    
    return image, label
