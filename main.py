# --- Train and Evaluate Utility ---
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def train_and_eval(model, trainloader, testloader, name, results_folder, epochs=10, lr=0.001, use_scheduler=False, validation_intervals=1, use_data_augmentation=True, continue_training=False, previous_epochs=0, balance_dataset_flag=True, step_size=5, gamma=0.5, loss_target=0, idx_to_species=None, df_lookup=None, train_df_lookup=None):
  """
  Trains the model, evaluates on validation data, and saves metrics and heatmap images to results folder.
  Uses eval_metrics and get_metrics logic from eval_utils.py.
  """
  # Train
  train(
    model, trainloader, epochs=epochs, name=name, loss_target=loss_target, lr=lr,
    use_scheduler=use_scheduler, validation_loader=testloader, validation_intervals=validation_intervals,
    use_data_augmentation=use_data_augmentation, continue_training=continue_training, previous_epochs=previous_epochs,
    balance_dataset_flag=balance_dataset_flag, step_size=step_size, gamma=gamma
  )

  # Evaluate using eval_metrics from eval_utils
  from eval_utils import eval_metrics
  if idx_to_species is None:
      raise ValueError("idx_to_species must be provided for metrics and heatmap labeling.")
  if df_lookup is None:
      raise ValueError("df_lookup must be provided for eval_metrics.")

  # --- Validation metrics and heatmap ---
  metrics, confusion_matrix = eval_metrics(model, testloader, df_lookup, idx_to_species, device)
  results_dir = os.path.join(os.getcwd(), "17", "results", results_folder)
  os.makedirs(results_dir, exist_ok=True)
  import pandas as pd
  overall_accuracy = None
  if 'all' in metrics and 'accuracy' in metrics['all']:
      overall_accuracy = metrics['all']['accuracy']
  metrics_rows = []
  for species, vals in metrics.items():
      if species == 'all':
          continue
      row = {'Species': species}
      for k in ['precision', 'recall', 'f1']:
          v = vals.get(k, None)
          row[k.capitalize()] = f"{v:.4f}" if v is not None else "-"
      metrics_rows.append(row)
  metrics_df = pd.DataFrame(metrics_rows)
  for col in ['Total', 'Accuracy']:
      if col in metrics_df.columns:
          metrics_df = metrics_df.drop(columns=[col])
  fig, ax = plt.subplots(figsize=(min(12, 2+len(metrics_df)*0.5), 1.2+len(metrics_df)*0.4))
  ax.axis('off')
  if overall_accuracy is not None:
      ax.text(0.5, 1.08, f"Overall Accuracy: {overall_accuracy*100:.2f}%", fontsize=14, ha='center', va='center', transform=ax.transAxes)
  tbl = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(12)
  tbl.scale(1.2, 1.2)
  plt.title(f"Validation Metrics: {name}", fontsize=14, pad=20)
  metrics_path = os.path.join(results_dir, f"validation_metrics_{name}.png")
  plt.savefig(metrics_path, bbox_inches='tight')
  plt.close(fig)
  heatmap_path = os.path.join(results_dir, f"validation_heatmap_{name}.png")
  print_heatmap_normalized(confusion_matrix, idx_to_species, heatmap_path)
  print(f"Validation metrics saved to {metrics_path}")
  print(f"Validation heatmap saved to {heatmap_path}")

  # --- Training metrics and heatmap ---
  # Use trainloader and train_df_lookup directly
  if train_df_lookup is not None:
      train_metrics, train_confusion_matrix = eval_metrics(model, trainloader, train_df_lookup, idx_to_species, device)
      train_overall_accuracy = None
      if 'all' in train_metrics and 'accuracy' in train_metrics['all']:
          train_overall_accuracy = train_metrics['all']['accuracy']
      train_metrics_rows = []
      for species, vals in train_metrics.items():
          if species == 'all':
              continue
          row = {'Species': species}
          for k in ['precision', 'recall', 'f1']:
              v = vals.get(k, None)
              row[k.capitalize()] = f"{v:.4f}" if v is not None else "-"
          train_metrics_rows.append(row)
      train_metrics_df = pd.DataFrame(train_metrics_rows)
      for col in ['Total', 'Accuracy']:
          if col in train_metrics_df.columns:
              train_metrics_df = train_metrics_df.drop(columns=[col])
      fig, ax = plt.subplots(figsize=(min(12, 2+len(train_metrics_df)*0.5), 1.2+len(train_metrics_df)*0.4))
      ax.axis('off')
      if train_overall_accuracy is not None:
          ax.text(0.5, 1.08, f"Overall Accuracy: {train_overall_accuracy*100:.2f}%", fontsize=14, ha='center', va='center', transform=ax.transAxes)
      tbl = ax.table(cellText=train_metrics_df.values, colLabels=train_metrics_df.columns, loc='center', cellLoc='center')
      tbl.auto_set_font_size(False)
      tbl.set_fontsize(12)
      tbl.scale(1.2, 1.2)
      plt.title(f"Training Metrics: {name}", fontsize=14, pad=20)
      train_metrics_path = os.path.join(results_dir, f"training_metrics_{name}.png")
      plt.savefig(train_metrics_path, bbox_inches='tight')
      plt.close(fig)
      train_heatmap_path = os.path.join(results_dir, f"training_heatmap_{name}.png")
      print_heatmap_normalized(train_confusion_matrix, idx_to_species, train_heatmap_path)
      print(f"Training metrics saved to {train_metrics_path}")
      print(f"Training heatmap saved to {train_heatmap_path}")
    
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import pandas as pd
from PIL import Image
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
import ast
import os
from eval_utils import *
from AnimalDataset import AnimalDataset
from generate_model_results import generate_model_results
from model import load_model, get_transform, get_df, device, get_pretrained_model, train, evaluate, print_image, split
from heatmap import print_heatmap, print_heatmap_normalized
from plot_accuracy_trends import plot_accuracy_trends, plot_entropy_scatter_simple


def xix(*args):
  return os.path.join(os.getcwd(), "19", *args)

def xvii(*args):
  return os.path.join(os.getcwd(), "17", *args)

df, species_to_idx = get_df(xvii("dataframes","ac_consensus_10_balanced_700.csv"))
transform = get_transform(width=300, crop_bottom=66, crop_left=55, crop_top=35)
dataset = AnimalDataset(df, img_dir=xvii("ac_consensus"), transform=transform)

train_test_split_ratio = 0.9
train_size = int(train_test_split_ratio * len(dataset))
test_size = len(dataset) - train_size

g_split = torch.Generator().manual_seed(5)
g_dataloader = torch.Generator().manual_seed(5)
train_indices, test_indices = split(dataset, train_test_split_ratio, generator=g_split)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=45, pin_memory=True, generator=g_dataloader, persistent_workers=True, prefetch_factor=4)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=45, pin_memory=True, generator=g_dataloader, persistent_workers=True, prefetch_factor=4)

df_lookup = dataset.df.iloc[test_dataset.indices].reset_index(drop=True)
train_df_lookup = dataset.df.iloc[train_dataset.indices].reset_index(drop=True)
idx_to_species = {idx: species for species, idx in species_to_idx.items()}

# TODO: experiments: data augmentation, pretrained vs not, 66% vs 100% consensus

#tmux: training
# model = resnet50(weights=None)
# model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(model.fc.in_features, len(species_to_idx))
# model = model.to(device)

# train_and_eval(
#   model,
#   trainloader,
#   testloader,
#   name="17_np_100groundtruth_30e_lrs-0005-5-07_augmented-nc_w300_rn50",
#   results_folder="np_100groundtruth_lrs-0005-5-07_augmented-nc_w300_rn50",
#   epochs=30,
#   lr=0.005,
#   use_scheduler=True,
#   validation_intervals=1,
#   use_data_augmentation=True,
#   continue_training=False,
#   previous_epochs=0,
#   balance_dataset_flag=True,
#   step_size=5,
#   gamma=0.7,
#   loss_target=0,
#   idx_to_species=idx_to_species,
#   df_lookup=df_lookup,
#   train_df_lookup=train_df_lookup
# )


# tmux: moretraining
# model = resnet18(weights=None)
# model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(model.fc.in_features, len(species_to_idx))
# model = model.to(device)

# train(model, trainloader, epochs=12, name="17_np_100groundtruth_12e_lrs-0005-5-05_augmented-nc_w256", use_scheduler=True, validation_loader=testloader, validation_intervals=1, lr=0.005, use_data_augmentation=True)


# model = load_model(xvii("models", "17_np_100groundtruth_36e_lrs-0005-5-05_augmented-nc_w256.pth"), species_to_idx, weights=None)
# train(model, trainloader, epochs=2, name="17_np_100groundtruth_26e_lrs-0005-5-05_augmented-nc_w256", use_scheduler=True, validation_loader=testloader, validation_intervals=1, lr=0.0003125, use_data_augmentation=True, continue_training=True, previous_epochs=24)

# Example of how to generate model results and create entropy analysis
from generate_model_results import generate_model_results

vs_test_data = pd.read_csv(xvii("dataframes", "ac_verified_0j.csv"))
vs_test_data["TrueSpecies"] = vs_test_data["TrueSpecies"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
vs_test_data["label"] = vs_test_data["TrueSpecies"].apply(lambda x: x[0]).map(species_to_idx)
vs_test_data["Classifications"] = vs_test_data["Classifications"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

subset_size = int(len(vs_test_data) * 1 ) # can edit number to just take random fraction of expert verified dataset
vs_test_data_subset = vs_test_data.sample(n=subset_size, random_state=42).reset_index(drop=True)
print(f"Using subset: {len(vs_test_data_subset)} samples out of {len(vs_test_data)}")

vs_test_dataset = AnimalDataset(vs_test_data_subset, img_dir=xvii("ac_verified"), transform=transform)
vs_testloader = DataLoader(vs_test_dataset, batch_size=64, shuffle=True, pin_memory=True)

# Create a non-shuffled loader for predictions to maintain index alignment
vs_testloader_no_shuffle = DataLoader(vs_test_dataset, batch_size=64, shuffle=False, pin_memory=True)

from plot_accuracy_trends import plot_entropy_scatter_simple

model = load_model(xvii("models", "17_p_100groundtruth_15e_lrs-0002-5-07_augmented-nc_w300_rn50.pth"), species_to_idx, weights=None)

# Generate model results with entropy calculation
print("Generating model predictions and entropy analysis...")
entropy_results = generate_model_results(model, vs_testloader_no_shuffle, vs_test_dataset, idx_to_species)

print(f"Generated {len(entropy_results)} results")
print(f"Overall accuracy: {sum(result[1] for result in entropy_results) / len(entropy_results):.3f}")

# Create the entropy analysis plot
print("Creating entropy vs accuracy plot...")
analysis = plot_entropy_scatter_simple(entropy_results)