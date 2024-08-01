from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy
import os
import PIL
from PIL import Image
import torch
# from libs import *

class PatientDataset(Dataset):
    def __init__(self, images_dir, classes, transform=None, target_transform=None):

        self.images_dir = images_dir
        self.classes = classes#["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
        self.target_transform = target_transform
        self.transform = transform

        self.img_paths = []
        self.img_class = []

        for classs in self.classes:
          for img_file in sorted(os.listdir(os.path.join(self.images_dir,classs))):
              self.img_paths.append(os.path.join(self.images_dir,classs,img_file))
              self.img_class.append(classs)


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        classs_name = self.img_class[idx]
        label = self.classes.index(classs_name)
        return image, label