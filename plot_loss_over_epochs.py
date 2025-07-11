import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

all_batch_losses = []

with open("batch_losses_zzid_resnet18_pretrained.csv", "r") as f:
  content = f.read()
  all_batch_losses = ast.literal_eval(content)

def plot(all_batch_losses,trainloader):
  batches_per_epoch = len(trainloader)
  epoch_numbers = np.repeat(range(1, 6), batches_per_epoch)[:len(all_batch_losses)]
  batch_in_epoch = np.arange(1, batches_per_epoch + 1).tolist() * 5
  x_values = epoch_numbers + (np.array(batch_in_epoch)[:len(all_batch_losses)] - 1) / batches_per_epoch


  plt.figure(figsize=(12, 6))
  plt.plot(x_values, all_batch_losses, color='blue', alpha=0.3, label='Raw loss')
  plt.plot(x_values, smooth(all_batch_losses, 50), color='red', label='Smoothed loss (window=50)')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss over Epochs')
  plt.legend()
  plt.grid(True)
  plt.show()

def smooth(y, box_pts):
  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode='same')
  return y_smooth