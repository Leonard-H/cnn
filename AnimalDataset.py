from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class AnimalDataset(Dataset):
  def __init__(self, df, img_dir, transform=None):
    self.df = df
    self.img_dir = Path(img_dir)
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self,idx):
    row = self.df.iloc[idx]
    img_path = self.img_dir / (str(row["subject_id"]) + ".jpg")
    image = Image.open(img_path).convert("RGB")
    label = row["label"]
    if self.transform: image = self.transform(image)
    return image, label