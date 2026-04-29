import torch
import torchvision
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.optim.lr_scheduler import StepLR 
from dataload import MP4Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import logging
from model import TwoTransformerClassifier
import argparse
from plot import plot_and_save
import warnings
warnings.simplefilter('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = torch.load('save/T2VSynthesis/model/best_model.pt')
sample_path = r'data\npy\Aphantasia\test\0\A_bird_building_a_nest_0.npy'
model.eval() 
total_test_loss = 0.0
with torch.no_grad():  
    data=np.load(sample_path)
    output = model(torch.unsqueeze(torch.tensor(data),dim=0))
    print(output)
