import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Plot loss of the batches during the training process. 
# At the end of each epoch, the validation set loss was also calculated, and those 
#   points are added here as well, to help identify overfitting.

all_batch_losses = []
validation_losses = []

with open("17/training_data/batch_losses_17_p_100groundtruth_12e_lrs-0002-5-07_augmented-nc_w300_rn50.csv", "r") as f:
  content = f.read()
  all_batch_losses += ast.literal_eval(content)
  
with open("17/training_data/validation_losses_17_p_100groundtruth_12e_lrs-0002-5-07_augmented-nc_w300_rn50.csv", "r") as f:
  content = f.read()
  validation_losses += ast.literal_eval(content)

def plot(all_batch_losses, trainloader, name, x_length=None, points=[]):
  batches_per_epoch = len(trainloader)
  max_epoch = 36
  total_batches = batches_per_epoch * max_epoch

  # Determine effective x-axis length
  if x_length is None:
    effective_len = len(all_batch_losses)
  else:
    effective_len = int(x_length * batches_per_epoch)

  # Pad or crop the loss data
  if len(all_batch_losses) < effective_len:
    padded_losses = all_batch_losses + [np.nan] * (effective_len - len(all_batch_losses))
  else:
    padded_losses = all_batch_losses[:effective_len]

  # Create x-values
  epoch_numbers = np.repeat(np.arange(max_epoch), batches_per_epoch)[:effective_len]
  batch_indices = np.tile(np.arange(batches_per_epoch), max_epoch)[:effective_len]
  x_values = epoch_numbers + batch_indices / batches_per_epoch

  # Plotting
  plt.figure(figsize=(12, 6))
  plt.plot(x_values, padded_losses, color='blue', alpha=0.3, label='Raw loss')

  window_size = 50
  if np.count_nonzero(~np.isnan(padded_losses)) >= window_size:
    # only smooth non-nan values
    clean_losses = np.array(padded_losses)
    valid_idx = ~np.isnan(clean_losses)
    smoothed = smooth(clean_losses[valid_idx], window_size)

    # Match x-values to smoothed
    smooth_x = np.array(x_values)[valid_idx][:len(smoothed)]
    plt.plot(smooth_x, smoothed, color='red', label=f'Smoothed loss (window={window_size})')

  # Add custom points as X markers
  if points:
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, marker='x', s=100, color='black', linewidth=2, label='Validation Set loss')

  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(name or 'Training Loss over Epochs')
  plt.legend()
  plt.grid(True)

  # Set fixed x-axis limits
  if x_length is not None:
    plt.xlim(0, x_length)
  else:
    plt.xlim(0, min(max_epoch, (len(all_batch_losses) / batches_per_epoch)))

  plt.show()
  plt.savefig(f"loss_plot_{name}.png")

def smooth(y, box_pts):
  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode='same')
  return y_smooth