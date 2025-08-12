import json
import pandas as pd
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50
import torch.optim as optim
import os
import ast
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if torch.cuda.is_available():
  print(f"GPU Name: {torch.cuda.get_device_name(device)}")
  print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
  print(torch.version.cuda)
  print(torch.backends.cudnn.version())    

def get_df(path):
  df = pd.read_csv(path)
  df["species"] = df["TrueSpecies"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
  species_to_idx = {species: idx for idx, species in enumerate(sorted(df["species"].unique()))}
  df["label"] = df["species"].map(species_to_idx)
  return (df, species_to_idx)


def load_model(path, species_to_idx, weights=None):
  loaded_model = resnet50(weights=weights)
  num_ftrs = loaded_model.fc.in_features
  loaded_model.fc = nn.Linear(num_ftrs, len(species_to_idx))
  loaded_model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
  loaded_model = loaded_model.to(device)
  
  return loaded_model


def get_transform(width, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
  original_width = 2688
  original_height = 1512
  
  cropped_width = original_width - crop_left - crop_right
  cropped_height = original_height - crop_top - crop_bottom
  
  height = int(width * (cropped_height / cropped_width))
  
  transform_list = []
  
  if crop_left or crop_right or crop_top or crop_bottom:
    def crop_sides(img):
      w, h = img.size
      left = crop_left
      top = crop_top
      right = w - crop_right
      bottom = h - crop_bottom
      return img.crop((left, top, right, bottom))
    
    transform_list.append(transforms.Lambda(crop_sides))

  transform_list.extend([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  transform = transforms.Compose(transform_list)
  
  return transform

def split(dataset, train_ratio, generator=None):
  if generator is None: generator = torch.Generator()
  
  species_groups = {}
  for idx in range(len(dataset)):
    species = dataset.df.iloc[idx]['species']
    if species not in species_groups:
      species_groups[species] = []
    species_groups[species].append(idx)
  
  train_indices = []
  test_indices = []
  
  for species, indices in species_groups.items():
    species_indices = torch.tensor(indices)
    shuffled_indices = species_indices[torch.randperm(len(species_indices), generator=generator)]
    
    n_train = int(len(indices) * train_ratio)
    
    train_indices.extend(shuffled_indices[:n_train].tolist())
    test_indices.extend(shuffled_indices[n_train:].tolist())
  
  return train_indices, test_indices

def get_pretrained_model(species_to_idx):
  m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  m.fc = nn.Linear(m.fc.in_features, len(species_to_idx))
  return m.to(device)


def train(m, loader, epochs, name, loss_target=0, lr=0.001, use_scheduler=False, validation_loader=None, validation_intervals=3, use_data_augmentation=True, continue_training=False, previous_epochs=0, balance_dataset_flag=True, step_size=5, gamma=0.5):
  if balance_dataset_flag:
    print("Balancing dataset...")
    balanced_dataset = balance_dataset(loader.dataset)
  else:
    print("Skipping dataset balancing for faster training...")
    balanced_dataset = loader.dataset
  
  if use_data_augmentation:
    print("Applying general data augmentation...")
    augmented_transform = get_augmented_transform()
    
    # Create new dataset with augmented transforms
    if hasattr(balanced_dataset, 'dataset'):  # If it's a Subset
      original_dataset = balanced_dataset.dataset
      augmented_dataset = AugmentedDataset(original_dataset, balanced_dataset.indices, augmented_transform)
    else:
      augmented_dataset = AugmentedDataset(balanced_dataset, None, augmented_transform)
    
    # Create new dataloader with augmented dataset
    loader = DataLoader(
      augmented_dataset, 
      batch_size=loader.batch_size, 
      shuffle=True, 
      num_workers=loader.num_workers, 
      pin_memory=loader.pin_memory
    )
    print(f"Dataset balanced and augmented. New size: {len(augmented_dataset)}")
  else:
    # Only balancing, no additional augmentation
    loader = DataLoader(
      balanced_dataset, 
      batch_size=loader.batch_size, 
      shuffle=True, 
      num_workers=loader.num_workers, 
      pin_memory=loader.pin_memory
    )
    print(f"Dataset balanced. New size: {len(balanced_dataset)}")
  
  # Training loop 
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(m.parameters(), lr=lr)
  
  # Enable mixed precision for speed
  scaler = torch.cuda.amp.GradScaler()
  
  if use_scheduler:
    from torch.optim.lr_scheduler import StepLR
    if continue_training:
      print(f"Continuing training from epoch {previous_epochs + 1}")
      print(f"Current step size: {step_size}")
      print(f"Note: StepLR will step every {step_size} epochs from now. If you want to resume exactly, consider adjusting step_size or last_epoch.")
      scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
      scheduler.last_epoch = previous_epochs - 1
    else:
      scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

  all_batch_losses = []
  validation_losses = []
  
  # Add timing and GPU monitoring
  import time
  epoch_start_time = time.time()
  batch_times = []

  for epoch in range(epochs):
    m.train()
    epoch_batch_start = time.time()

    if use_scheduler and scheduler is not None:
      current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

      epochs_since_last = scheduler.last_epoch % step_size
      epochs_to_next = step_size - epochs_since_last
      print(f"[Epoch {epoch+1}] Current learning rate: {current_lr:.6e}. Next reduction in {epochs_to_next} epoch(s).")
    else:
      print(f"[Epoch {epoch+1}] Current learning rate: {optimizer.param_groups[0]['lr']:.6e}.")

    for batch_idx, (images, labels) in enumerate(loader):
      batch_start_time = time.time()
      images, labels = images.to(device), labels.to(device)
      data_load_time = time.time() - batch_start_time

      optimizer.zero_grad()

      # Use mixed precision
      with torch.cuda.amp.autocast():
        outputs = m(images)
        loss = criterion(outputs, labels)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      batch_total_time = time.time() - batch_start_time
      batch_times.append((data_load_time, batch_total_time))

      # Print detailed timing every 50 batches
      if batch_idx % 50 == 0:
        gpu_util = torch.cuda.utilization(device) if torch.cuda.is_available() else 0
        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
        print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.4f}")
        print(f"  Data load time: {data_load_time:.3f}s, Total batch time: {batch_total_time:.3f}s")
        print(f"  GPU Util: {gpu_util}%, GPU Memory: {gpu_memory:.1f}GB")

      all_batch_losses.append(loss.item())

    # Handle scheduler stepping
    if use_scheduler and scheduler is not None:
      scheduler.step()

    # Validation and early stopping based on validation loss
    stop_training = False
    if ((epoch+1) % validation_intervals == 0) and validation_loader is not None:
      val_loss = evaluate(m, validation_loader)
      print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
      validation_losses.append((epoch+1+previous_epochs, val_loss))
      if val_loss <= loss_target:
        print(f"Early stopping: Validation loss {val_loss:.4f} <= loss_target {loss_target:.4f}")
        stop_training = True
    if stop_training:
      break

  model_save_path = os.path.join(os.getcwd(), "17", "models", name + ".pth")
  torch.save(m.state_dict(), model_save_path)
  print(f"Model saved to {model_save_path}")

  loss_save_path = os.path.join(os.getcwd(), "17", "training_data", f"batch_losses_{name}.csv")
  validation_loss_save_path = os.path.join(os.getcwd(), "17", "training_data", f"validation_losses_{name}.csv")
  
  with open(loss_save_path, 'w') as f:
    json.dump(all_batch_losses, f)
    print(f"Batch losses saved to {loss_save_path}")
    
  with open(validation_loss_save_path, 'w') as f:
    json.dump(validation_losses, f)
    print(f"Validation losses saved to {validation_loss_save_path}")  

def balance_dataset(dataset):
  """Phase 1: Balance dataset by duplicating samples from underrepresented classes"""
  from collections import Counter
  import random
  
  # Set random seed for reproducible sampling
  random.seed(42)
  
  # Get the underlying dataset and indices
  if hasattr(dataset, 'dataset'):
    base_dataset = dataset.dataset
    indices = dataset.indices
  else:
    base_dataset = dataset
    indices = list(range(len(dataset)))
  
  # Count samples per class
  class_counts = Counter()
  class_indices = {}
  
  for idx in indices:
    label = base_dataset.df.iloc[idx]['label']
    class_counts[label] += 1
    if label not in class_indices:
      class_indices[label] = []
    class_indices[label].append(idx)
  
  # Find the maximum count
  max_count = max(class_counts.values())
  print(f"Class distribution: {dict(class_counts)}")
  print(f"Target count per class: {max_count}")
  
  # Create balanced indices
  balanced_indices = []
  for label, current_indices in class_indices.items():
    current_count = len(current_indices)
    
    # Add all original indices
    balanced_indices.extend(current_indices)
    
    # Duplicate samples to reach max_count
    if current_count < max_count:
      needed = max_count - current_count
      # Randomly sample with replacement from existing indices
      additional_indices = random.choices(current_indices, k=needed)
      balanced_indices.extend(additional_indices)
      print(f"Class {label}: {current_count} -> {max_count} (added {needed} samples)")
  
  # Return a Subset with balanced indices
  from torch.utils.data import Subset
  return Subset(base_dataset, balanced_indices)

def get_augmented_transform():
  torch.manual_seed(42)
  
  return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # ColorJitter seems to make the model less accurate
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
  ])

class AugmentedDataset(torch.utils.data.Dataset):
  """Dataset wrapper that applies additional augmentation transforms"""
  def __init__(self, base_dataset, indices=None, augment_transform=None):
    self.base_dataset = base_dataset
    self.indices = indices if indices is not None else list(range(len(base_dataset)))
    self.augment_transform = augment_transform
  
  def __len__(self):
    return len(self.indices)
  
  def __getitem__(self, idx):
    actual_idx = self.indices[idx]
    
    # Get the original image path and label
    if hasattr(self.base_dataset, 'dataset'):  # If base is also a Subset
      subject_id = self.base_dataset.dataset.df.iloc[actual_idx]['subject_id']
      img_path = str(subject_id) + ".jpg"
      label = self.base_dataset.dataset.df.iloc[actual_idx]['label']
      img_dir = self.base_dataset.dataset.img_dir
    else:
      subject_id = self.base_dataset.df.iloc[actual_idx]['subject_id']
      img_path = str(subject_id) + ".jpg"
      label = self.base_dataset.df.iloc[actual_idx]['label']
      img_dir = self.base_dataset.img_dir
    
    # Load and process image
    from PIL import Image
    image = Image.open(os.path.join(img_dir, img_path)).convert('RGB')
    
    # Apply augmentation if provided (before the base transform)
    if self.augment_transform:
      image = self.augment_transform(image)
    
    # Apply the base transform (this should include ToTensor and Normalize)
    if hasattr(self.base_dataset, 'transform') and self.base_dataset.transform:
      image = self.base_dataset.transform(image)
    
    return image, label
    
def evaluate(m, loader):
  criterion = nn.CrossEntropyLoss()
  m.eval()
  total_loss = 0.0
  n_batches = 0
  
  with torch.no_grad():
    for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      outputs = m(images)
      loss = criterion(outputs, labels)
      total_loss += loss.item()
      n_batches += 1
      
  return total_loss / n_batches if n_batches > 0 else float('inf')


def print_image(img_path, df, transform=None):
  """Function visually displays tensor to check that metadata was properly cropped out, and to get an idea of how much the resultion is reduced."""
  try:
    image = Image.open(img_path).convert('RGB')
    print(f"Found image: {img_path}")
    
    # Apply transform if provided
    if transform:
      tensor_image = transform(image)
      print(f"Tensor shape after transform: {tensor_image.shape}")
      
      # Ensure tensor is 3D (C, H, W) for ToPILImage
      if len(tensor_image.shape) != 3:
          raise ValueError(f"Unexpected tensor shape: {tensor_image.shape}")
      
      # Denormalize if using standard normalization
      tensor_denorm = tensor_image * 0.5 + 0.5  # Reverse normalize
      tensor_denorm = torch.clamp(tensor_denorm, 0, 1)
      
      # Convert to PIL
      to_pil = transforms.ToPILImage()
      image_to_save = to_pil(tensor_denorm)
    else:
      image_to_save = image
    
    # Display the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_to_save)
    plt.axis('off')
    plt.show()
    
    # Save to example.jpg
    image_to_save.save('example.jpg')
    print("Image saved to example.jpg")
        
  except FileNotFoundError:
    print(f"Image file not found: {img_path}")
  except ValueError as ve:
    print(f"ValueError: {ve}")
  except Exception as e:
    print(f"Error loading image: {e}")