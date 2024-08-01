### IMPORT LIBRARIES ###

import argparse
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from data_load import PatientDataset 
import torchvision
# from autoaugment import ImageNetPolicy

from torchvision.models import (resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, 
                                vit_l_32, ViT_L_32_Weights, vit_l_16, ViT_L_16_Weights, 
                                maxvit_t, MaxVit_T_Weights, swin_b, Swin_B_Weights,
                                efficientnet_v2_m, EfficientNet_V2_M_Weights)
from scipy.stats import spearmanr#, multipletests
from sklearn.metrics import auc
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix