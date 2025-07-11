import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
import ast
import os
from eval_utils import eval_metrics, eval_volunteer_metrics, display_metrics
from AnimalDataset import AnimalDataset
from model import load_model, get_transform, get_df, device, get_pretrained_model, train
from heatmap import print_heatmap

df, species_to_idx = get_df(os.path.join(os.getcwd(), "zzid_train.csv"))

# create transform, retaining original aspect ratio
transform = get_transform(width=256)
dataset = AnimalDataset(df, img_dir=os.path.join(os.getcwd(), "zzid_train"), transform=transform)

train_test_split_ratio = 0.8
train_size = int(train_test_split_ratio * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(5)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

subset_size = int(1 * len(test_dataset))
test_subset_indices = test_dataset.indices[:subset_size]
test_subset = torch.utils.data.Subset(dataset, test_subset_indices)

trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2 ,pin_memory=True)
testloader_subset = DataLoader(test_subset, batch_size=256, shuffle=False, pin_memory=True)

loaded_model = load_model(os.path.join(os.getcwd(), "zzid_resnet18_not_pretrained.pth"), species_to_idx)

df_lookup = dataset.df.iloc[test_subset.indices].reset_index(drop=True)
idx_to_species = {idx: species for species, idx in species_to_idx.items()}


# uncomment lines to run them

# metrics, confusion_matrix = eval_metrics(loaded_model, testloader_subset, df_lookup, idx_to_species, device)

###---

# pretrained_model = get_pretrained_model(species_to_idx)
# train(pretrained_model, trainloader, epochs=5, name="zzid_resnet18_pretrained", loss_target=0.29)

# pretrained_model = load_model(os.path.join(os.getcwd(), "zzid_resnet18_pretrained.pth"), species_to_idx)
# metrics, confusion_matrix = eval_metrics(pretrained_model, testloader_subset, df_lookup, idx_to_species, device)
# display_metrics(metrics)
# print_heatmap(confusion_matrix, idx_to_species)
