import ast
from collections import defaultdict
import torch
import pandas as pd


def get_metrics(stats):
  metrics = {}

  for species, values in stats.items():
    acc = values["correct"] / values["total"] if values["total"] > 0 else None

    if species == "all":
      metrics[species] = {"accuracy": acc}
    else:
      TP, FP, FN = values["TP"], values["FP"], values["FN"]

      precision = TP / (TP + FP) if (TP + FP) > 0 else None
      recall = TP / (TP + FN) if (TP + FN) > 0 else None
      f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
      )

      metrics[species] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
      }
  
  return metrics

def eval_metrics(model, dataloader, df_lookup, idx_to_species, device):
  model.eval()

  stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "correct": 0, "total": 0})

  # confusion matrix
  num_species = len(idx_to_species)
  confusion_matrix = torch.zeros(num_species, num_species)

  with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader):
      print(f"Batch {batch_idx}/{len(dataloader)}")
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      for i in range(len(labels)):
        true_idx = labels[i].item()
        pred_idx = predicted[i].item()

        confusion_matrix[true_idx, pred_idx] += 1

        row = df_lookup.iloc[batch_idx * dataloader.batch_size + i]
        true_species = row["species"]
        pred_species = idx_to_species[pred_idx]

        stats["all"]["correct"] += int(pred_idx == true_idx)
        stats["all"]["total"] += 1

        stats[true_species]["correct"] += int(pred_idx == true_idx)
        stats[true_species]["total"] += 1

        if true_idx == pred_idx:
          stats[true_species]["TP"] += 1
        else:
          stats[pred_species]["FP"] += 1
          stats[true_species]["FN"] += 1
  
  return get_metrics(stats), confusion_matrix



def eval_volunteer_metrics(df_lookup):
  stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "correct": 0, "total": 0})

  for index, row in df_lookup.iterrows():
      true_species = row["species"]

      cls = ast.literal_eval(row["Classifications"])
      for cl in cls:
        stats["all"]["correct"] += int(cl == true_species)
        stats["all"]["total"] += 1

        stats[true_species]["correct"] += int(cl == true_species)
        stats[true_species]["total"] += 1

        if cl == true_species:
          stats[true_species]["TP"] += 1
        else:
          stats[cl]["FP"] += 1
          stats[true_species]["FN"] += 1

  return get_metrics(stats)

def display_metrics(metrics):
  metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

  for col in ['accuracy', 'precision', 'recall', 'f1']:
    if col in metrics_df.columns:
      metrics_df[col] = metrics_df[col].map(
        lambda x: f'{x * 100:.1f}%' if pd.notnull(x) else ''
      )

  return metrics_df