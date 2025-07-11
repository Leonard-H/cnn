import json
import pandas as pd
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_df(path):
  df = pd.read_csv(path)
  species_to_idx = {species: idx for idx, species in enumerate(sorted(df["species"].unique()))}
  df["label"] = df["species"].map(species_to_idx)
  return (df, species_to_idx)


def load_model(path, species_to_idx):
  loaded_model = resnet18(weights=None)
  num_ftrs = loaded_model.fc.in_features
  loaded_model.fc = nn.Linear(num_ftrs, len(species_to_idx))
  loaded_model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
  loaded_model = loaded_model.to(device)
  return loaded_model


def get_transform(width):
  height = int(width / (2688 / 1512))

  transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  return transform

def get_pretrained_model(species_to_idx):
  m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  m.fc = nn.Linear(m.fc.in_features, len(species_to_idx))
  return m.to(device)


def train(m, loader, epochs, name, loss_target=0):
  # does not return anything, but saves model and losses to path
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(m.parameters(), lr=0.001)

  all_batch_losses = []
  n_under_loss_target = 0

  for epoch in range(epochs):
    if n_under_loss_target >= 10: break
    m.train()
    for images, labels in loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = m(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.4f}")
      all_batch_losses.append(loss.item())

      if loss.item() <= loss_target: n_under_loss_target += 1
      if n_under_loss_target >= 10: break

  model_save_path = os.path.join(os.getcwd(), name + ".pth")
  torch.save(m.state_dict(), model_save_path)
  print(f"Model saved to {model_save_path}")

  loss_save_path = os.path.join(os.getcwd(), f"batch_losses_{name}.csv")
  with open(loss_save_path, 'w') as f:
    json.dump(all_batch_losses, f)
  print(f"Batch losses saved to {loss_save_path}")