import ast
from collections import defaultdict
import torch
import pandas as pd


def get_metrics(stats):
  metrics = {}

  for species, values in stats.items():
    if species == "all":
      # Overall accuracy is still meaningful (total correct / total predictions)
      accuracy = values["correct"] / values["total"] if values["total"] > 0 else None
      metrics[species] = {
        "accuracy": accuracy,
        "total": values["total"]
      }
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": values["total"]
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
      print(f"Batch {batch_idx+1}/{len(dataloader)}")
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      for i in range(len(labels)):
        true_idx = labels[i].item()
        pred_idx = predicted[i].item()

        confusion_matrix[true_idx, pred_idx] += 1

        # Use idx_to_species mapping for both true and predicted species
        true_species = idx_to_species[true_idx]
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
  # Extract overall accuracy if it exists
  overall_accuracy = None
  if 'all' in metrics and 'accuracy' in metrics['all']:
    overall_accuracy = metrics['all']['accuracy']
  
  # Create DataFrame excluding 'all' row, sorted alphabetically by species name
  species_metrics = {k: v for k, v in metrics.items() if k != 'all'}
  
  # If no species metrics, return empty DataFrame
  if not species_metrics:
    if overall_accuracy is not None:
      print(f"Overall accuracy: {overall_accuracy * 100:.1f}%")
      print()
    return pd.DataFrame()
  
  # Sort species alphabetically before creating DataFrame
  sorted_species_metrics = dict(sorted(species_metrics.items()))
  metrics_df = pd.DataFrame.from_dict(sorted_species_metrics, orient='index')
  
  # Keep only the three main columns and format as percentages
  columns_to_keep = ['recall', 'precision', 'f1']
  available_columns = [col for col in columns_to_keep if col in metrics_df.columns]
  
  for col in available_columns:
    metrics_df[col] = metrics_df[col].map(
      lambda x: f'{x * 100:.1f}%' if pd.notnull(x) else ''
    )
  
  # Keep only the available desired columns
  if available_columns:
    metrics_df = metrics_df[available_columns]
  
  # Print overall accuracy at the top
  if overall_accuracy is not None:
    print(f"Overall accuracy: {overall_accuracy * 100:.1f}%")
    print()
  
  return metrics_df





# metrics for test data in verification set
# to find correlation between volunteers finding an image difficult to classify vs. the model getting it right

def create_species_divisions_from_classifications(species, classifications_str):
  """
  Create species divisions by calculating n_different and n_wrong from Classifications string.
  
  Args:
    species: The true species for this image
    classifications_str: String representation of volunteer classifications
    
  Returns:
    List of division names for this species/image
  """
  from plot_accuracy_trends import flatten_verified_classifications
  from collections import Counter
  
  # Parse the classifications string
  try:
    classifications = ast.literal_eval(classifications_str)
  except:
    # Fallback if parsing fails
    return [species, 'all']
  
  # Flatten the classifications
  flattened = flatten_verified_classifications(species, classifications)
  
  # Calculate metrics
  n_different = len(set(flattened)) if flattened else 0
  n_wrong = sum(1 for c in flattened if c != species) if flattened else 0
  
  # Create divisions
  divisions = [
    species,
    f"{species}_n_different_{n_different}",
    f"{species}_n_wrong_{n_wrong}",
    'all'
  ]
  
  return divisions

def merge_metrics(metrics):
  aggregated = {
    'overall': metrics['all'],
    'by_species': {},
    'by_n_different': defaultdict(list),
    'by_n_wrong': defaultdict(list),
    'divisions': {}
  }

  aggregated['overall']['count'] = metrics['all'].get('total', 0)

  for key, values in metrics.items():
    if key == 'all': continue
    elif '_n_different_' in key:
      base_species, n = key.split('_n_different_')
      n = int(n)
      values_with_count = values.copy()
      values_with_count['count'] = values.get('total', 0)
      aggregated['by_n_different'][n].append(values_with_count)
      aggregated['divisions'][key] = values_with_count
    elif '_n_wrong_' in key:
      base_species, n = key.split('_n_wrong_')
      n = int(n)
      values_with_count = values.copy()
      values_with_count['count'] = values.get('total', 0)
      aggregated['by_n_wrong'][n].append(values_with_count)
      aggregated['divisions'][key] = values_with_count
    elif '_' not in key:
      values_with_count = values.copy()
      values_with_count['count'] = values.get('total', 0)
      aggregated['by_species'][key] = values_with_count
      aggregated['divisions'][key] = values_with_count,

  for category in ['by_n_different', 'by_n_wrong']:
    # First calculate counts
    count_by_n = {
      n: sum(v['count'] for v in values)
      for n, values in aggregated[category].items()
    }
    
    # Then calculate means for metrics
    aggregated[category] = {
      n: {
        metric: pd.DataFrame(values)[metric].mean()
        for metric in ['accuracy', 'precision', 'recall', 'f1']
        if not pd.DataFrame(values)[metric].isna().all()
      }
      for n, values in aggregated[category].items()
    }
    
    # Add counts back to the aggregated metrics
    for n in aggregated[category]:
      aggregated[category][n]['count'] = count_by_n[n]

  return aggregated

def eval_metrics_with_divisions(model, dataloader, df_lookup, idx_to_species, device):
  model.eval()

  stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "correct": 0, "total": 0})
  num_species = len(idx_to_species)
  confusion_matrix = torch.zeros(num_species, num_species)

  with torch.no_grad():
    sample_idx = 0  # Track position in df_lookup
    
    for batch_idx, (images, labels) in enumerate(dataloader):
      print(f"Batch {batch_idx+1}/{len(dataloader)}")
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      for i in range(len(labels)):
        true_idx = labels[i].item()
        pred_idx = predicted[i].item()

        confusion_matrix[true_idx, pred_idx] += 1

        # Use idx_to_species mapping for both true and predicted species
        true_species = idx_to_species[true_idx]
        pred_species = idx_to_species[pred_idx]

        # Get the corresponding row from df_lookup
        if sample_idx < len(df_lookup):
          row = df_lookup.iloc[sample_idx]
          
          # Create all divisions using the Classifications data
          classifications_str = row.get('Classifications', '[]')
          true_divisions = create_species_divisions_from_classifications(true_species, classifications_str)
          pred_divisions = create_species_divisions_from_classifications(pred_species, classifications_str)
        else:
          # Fallback if we don't have row data
          true_divisions = [true_species, 'all']
          pred_divisions = [pred_species, 'all']

        # Update stats for all divisions
        for division in true_divisions:
          stats[division]["correct"] += int(pred_idx == true_idx)
          stats[division]["total"] += 1

          if true_idx == pred_idx:
            stats[division]["TP"] += 1
          else:
            # Handle FP for predicted divisions
            for pred_div in pred_divisions:
              if pred_div != 'all' and pred_div != division:
                stats[pred_div]["FP"] += 1
            # Handle FN for true division
            if division != 'all':
              stats[division]["FN"] += 1

        sample_idx += 1

  metrics = get_metrics(stats)
  aggregated_metrics = merge_metrics(metrics)
  
  return aggregated_metrics, confusion_matrix




