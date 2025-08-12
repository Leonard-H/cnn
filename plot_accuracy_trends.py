import ast
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy
import numpy as np
import pandas as pd
from scipy import stats

# def plot_accuracy_trends(metrics, wrong_max=0, different_max=0):
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
  
#   wrong_x = sorted(metrics['by_n_wrong'].keys())
#   # Filter data based on wrong_max parameter
#   if wrong_max > 0:
#     wrong_x = [x for x in wrong_x if x <= wrong_max]
  
#   wrong_y = [metrics['by_n_wrong'][i]['accuracy'] for i in wrong_x]
#   wrong_samples = [metrics['by_n_wrong'][i]['count'] for i in wrong_x]
  
#   # Debug print
#   print(f"Wrong samples: {wrong_samples}")
#   print(f"Max wrong samples: {max(wrong_samples) if wrong_samples else 0}")
  
#   ax1_twin = ax1.twinx()
  
#   line1 = ax1.plot(wrong_x, wrong_y, 'bo-', label='Accuracy', zorder=3)
#   ax1.set_xlabel('Number of Wrong Classifications')
#   ax1.set_ylabel('Accuracy')
#   ax1.set_title('Model Accuracy vs Number of Wrong Classifications')
#   ax1.grid(True, alpha=0.3)
#   ax1.set_ylim(0, 1)
  
#   # Adjust bar width and center them on x positions
#   bar_width = 0.6
#   bars1 = ax1_twin.bar([x for x in wrong_x], wrong_samples, 
#                        width=bar_width, alpha=0.5, color='lightblue', label='Sample Size', zorder=1)
#   ax1_twin.set_ylabel('Sample Size')
#   # Force y-axis limits for sample size
#   max_wrong = max(wrong_samples) if wrong_samples else 100
#   ax1_twin.set_ylim(0, max_wrong * 1.2)
  
#   diff_x = sorted(metrics['by_n_different'].keys())
#   # Filter data based on different_max parameter
#   if different_max > 0:
#     diff_x = [x for x in diff_x if x <= different_max]
  
#   diff_y = [metrics['by_n_different'][i]['accuracy'] for i in diff_x]
#   diff_samples = [metrics['by_n_different'][i]['count'] for i in diff_x]
  
#   # Debug print
#   print(f"Diff samples: {diff_samples}")
#   print(f"Max diff samples: {max(diff_samples) if diff_samples else 0}")
  
#   ax2_twin = ax2.twinx()
  
#   line2 = ax2.plot(diff_x, diff_y, 'ro-', label='Accuracy', zorder=3)
#   ax2.set_xlabel('Number of Different Classifications')
#   ax2.set_ylabel('Accuracy')
#   ax2.set_title('Model Accuracy vs Number of Different Classifications')
#   ax2.grid(True, alpha=0.3)
#   ax2.set_ylim(0, 1)
  
#   # Adjust bar width and center them on x positions
#   bars2 = ax2_twin.bar([x for x in diff_x], diff_samples, 
#                        width=bar_width, alpha=0.5, color='lightcoral', label='Sample Size', zorder=1)
#   ax2_twin.set_ylabel('Sample Size')
#   # Force y-axis limits for sample size
#   max_diff = max(diff_samples) if diff_samples else 100
#   ax2_twin.set_ylim(0, max_diff * 1.2)
  
#   # Add legends
#   lines1, labels1 = ax1.get_legend_handles_labels()
#   lines2, labels2 = ax1_twin.get_legend_handles_labels()
#   ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
  
#   lines1, labels1 = ax2.get_legend_handles_labels()
#   lines2, labels2 = ax2_twin.get_legend_handles_labels()
#   ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
  
#   plt.tight_layout()
#   plt.savefig('accuracy_trends.png', dpi=300, bbox_inches='tight')
#   plt.show()
  
  
def flatten_verified_classifications(species, classifications):
  """
  Flatten a 2D array of classifications to a 1D array.
  
  Args:
    species: The true species for this image
    classifications: 2D array where each sub-array contains species identified by a volunteer
    
  Returns:
    1D array of strings, same length as classifications
  """
  import random
  from collections import Counter
  
  single_species = []
  for classification in classifications:
    if len(classification) == 1: single_species.append(classification[0])
  
  if single_species: most_common = Counter(single_species).most_common(1)[0][0]
  else: most_common = None
  
  result = []
  
  for classification in classifications:
    if len(classification) == 1:
      result.append(classification[0])
    elif len(classification) > 1:
      if species in classification: result.append(species)
      elif most_common and most_common in classification: result.append(most_common)
      else: result.append(random.choice(classification))
    else:
      result.append(None)
  
  return result

def get_shannon_entropy(subject):
  flattened_classifications = flatten_verified_classifications(subject["TrueSpecies"][0], subject["Classifications"])
  counts = Counter(flattened_classifications)
  probabilities = np.array(list(counts.values())) / len(flattened_classifications)
  return entropy(probabilities, base=2)

def plot_entropy_vs_accuracy(entropy_results, bin_method='equal_width', n_bins=5, upper_buffer=0):
  """
  Plot model accuracy vs Shannon entropy of volunteer classifications.
  
  Args:
    entropy_results: List of tuples (entropy, correct) from generate_model_results
    bin_method: 'equal_width', 'equal_freq', or 'custom'
    n_bins: Number of bins to use
  """
  
  # Extract entropies and accuracies from tuples
  entropies = np.array([result[0] for result in entropy_results])
  accuracies = np.array([result[1] for result in entropy_results])
  
  # Logistic regression: test if entropy predicts accuracy (binary outcome)
  try:
    import statsmodels.api as sm
    X_logit = sm.add_constant(entropies)
    model_logit = sm.Logit(accuracies, X_logit)
    result_logit = model_logit.fit(disp=0)
    coef = result_logit.params[1]
    pval = result_logit.pvalues[1]
    odds_ratio = np.exp(coef)
    print(f"\nLogistic regression: logit(accuracy) = {result_logit.params[0]:.4f} + {coef:.4f} * entropy")
    print(f"Logistic regression p-value for entropy: {pval:.6f}")
    print(f"Odds ratio for entropy: {odds_ratio:.4f}")
    if pval < 0.001:
      print("*** Highly significant (p < 0.001)")
    elif pval < 0.01:
      print("** Significant (p < 0.01)")
    elif pval < 0.05:
      print("* Significant (p < 0.05)")
    else:
      print("Not significant (p >= 0.05)")
  except ImportError:
    print("statsmodels is not installed. Install it with 'pip install statsmodels' to run logistic regression.")
  except Exception as e:
    print(f"Logistic regression failed: {e}")

  # For bins analysis, optionally exclude top upper_buffer fraction of subjects by entropy
  if upper_buffer > 0:
    cutoff = np.quantile(entropies, 1 - upper_buffer)
    bin_mask = entropies <= cutoff
    entropies_bins = entropies[bin_mask]
    accuracies_bins = accuracies[bin_mask]
  else:
    entropies_bins = entropies
    accuracies_bins = accuracies
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  # Plot 1: Scatter plot with linear regression line
  ax1.scatter(entropies, accuracies, alpha=0.6, s=30)
  
  # Add linear regression line
  slope, intercept, r_value, p_value, std_err = stats.linregress(entropies, accuracies)
  line_x = np.array([entropies.min(), entropies.max()])
  line_y = slope * line_x + intercept
  
  ax1.plot(line_x, line_y, 'r-', linewidth=2, label=f'r={r_value:.3f}, p={p_value:.3f}')
  ax1.set_xlabel('Shannon Entropy')
  ax1.set_ylabel('Model Accuracy (0/1)')
  ax1.set_title('Model Accuracy vs Shannon Entropy (Scatter)')
  ax1.grid(True, alpha=0.3)
  ax1.legend()
  
  # Plot 2: Binned analysis (optionally filtered by entropy_ceiling)
  if bin_method == 'equal_width':
    bins = np.linspace(entropies_bins.min(), entropies_bins.max(), n_bins + 1)
  elif bin_method == 'equal_freq':
    bins = np.percentile(entropies_bins, np.linspace(0, 100, n_bins + 1))
  else:  # custom bins
    bins = np.linspace(0, entropies_bins.max(), n_bins + 1)

  bin_centers = []
  bin_accuracies = []
  bin_counts = []
  bin_stds = []
  bin_labels = []  # Store labels for bins that actually have data

  for i in range(len(bins) - 1):
    mask = (entropies_bins >= bins[i]) & (entropies_bins < bins[i + 1])
    if i == len(bins) - 2:  # Include the last point
      mask = (entropies_bins >= bins[i]) & (entropies_bins <= bins[i + 1])

    if np.sum(mask) > 0:
      bin_centers.append((bins[i] + bins[i + 1]) / 2)
      bin_acc = accuracies_bins[mask]
      bin_accuracies.append(np.mean(bin_acc))
      bin_counts.append(len(bin_acc))
      bin_stds.append(np.std(bin_acc) / np.sqrt(len(bin_acc)))  # Standard error
      bin_labels.append(f'{bins[i]:.2f}-{bins[i+1]:.2f}')  # Only add label for bins with data


  ax2_twin = ax2.twinx()

  # Plot sample size as faded bars (background)
  bars = ax2_twin.bar(
      range(len(bin_centers)), bin_counts,
      width=0.7, alpha=0.25, color='gray', label='Sample Size', zorder=1)
  ax2_twin.set_ylabel('Sample Count')
  ax2_twin.legend(loc='upper right')

  # Plot accuracy as a line with dots and error bars (standard error)
  ax2.errorbar(
      range(len(bin_centers)), bin_accuracies, yerr=bin_stds,
      fmt='o-', color='C0', capsize=5, label='Mean Accuracy', zorder=2)
  ax2.set_xlabel('Entropy Bins')
  ax2.set_ylabel('Mean Model Accuracy')
  ax2.set_title('Model Accuracy by Entropy Bins')
  ax2.grid(True, alpha=0.3)

  # Set x-tick labels for bins that actually have data
  ax2.set_xticks(range(len(bin_centers)))
  ax2.set_xticklabels(bin_labels, rotation=45, ha='right')

  # Optionally, add legend for accuracy
  ax2.legend(loc='upper left')
  
  
  plt.tight_layout()
  plt.savefig('entropy_vs_accuracy.png', dpi=300, bbox_inches='tight')
  plt.show()
  
  # Print summary statistics
  print("\nEntropy vs Accuracy Analysis:")
  print(f"Entropy range: {entropies.min():.3f} - {entropies.max():.3f}")
  print(f"Overall accuracy: {np.mean(accuracies):.3f}")
  
  # Linear regression statistics
  print(f"Linear regression: y = {slope:.4f}x + {intercept:.4f}")
  print(f"Correlation coefficient (r): {r_value:.3f}")
  print(f"R-squared: {r_value**2:.3f}")
  print(f"P-value: {p_value:.6f}")
  if p_value < 0.001:
    print("*** Highly significant (p < 0.001)")
  elif p_value < 0.01:
    print("** Significant (p < 0.01)")
  elif p_value < 0.05:
    print("* Significant (p < 0.05)")
  else:
    print("Not significant (p >= 0.05)")
  
  return {
    'entropies': entropies,
    'accuracies': accuracies,
    'correlation': r_value,
    'r_squared': r_value**2,
    'p_value': p_value,
    'slope': slope,
    'intercept': intercept
  }
  
  


def plot_entropy_scatter_simple(entropy_results):
  """
  Simple scatter plot using entropy results tuples.
  
  Args:
    entropy_results: List of tuples (entropy, correct) from generate_model_results
  """
  # Extract entropies and accuracies from tuples
  entropies = np.array([result[0] for result in entropy_results])
  accuracies = np.array([result[1] for result in entropy_results])
  
  # Create single plot
  fig, ax = plt.subplots(1, 1, figsize=(8, 6))
  
  # Scatter plot
  ax.scatter(entropies, accuracies, alpha=0.6, s=30)
  
  # Add linear regression line
  slope, intercept, r_value, p_value, std_err = stats.linregress(entropies, accuracies)
  
  # Create line points
  line_x = np.array([entropies.min(), entropies.max()])
  line_y = slope * line_x + intercept
  
  ax.plot(line_x, line_y, 'r-', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')
  ax.set_xlabel('Shannon Entropy')
  ax.set_ylabel('Model Accuracy (0/1)')
  ax.set_title('Model Accuracy vs Shannon Entropy')
  ax.grid(True, alpha=0.3)
  ax.legend()
  
  plt.tight_layout()
  plt.savefig('entropy_scatter_simple.png', dpi=300, bbox_inches='tight')
  plt.show()
  
  # Print summary statistics
  print("\nEntropy vs Accuracy Analysis:")
  print(f"Entropy range: {entropies.min():.3f} - {entropies.max():.3f}")
  print(f"Overall accuracy: {np.mean(accuracies):.3f}")
  
  # Linear regression statistics
  print(f"Linear regression: y = {slope:.4f}x + {intercept:.4f}")
  print(f"Correlation coefficient (r): {r_value:.3f}")
  print(f"R-squared: {r_value**2:.3f}")
  print(f"P-value: {p_value:.6f}")
  if p_value < 0.001:
    print("*** Highly significant (p < 0.001)")
  elif p_value < 0.01:
    print("** Significant (p < 0.01)")
  elif p_value < 0.05:
    print("* Significant (p < 0.05)")
  else:
    print("Not significant (p >= 0.05)")
  
  return {
    'entropies': entropies,
    'accuracies': accuracies,
    'correlation': r_value,
    'r_squared': r_value**2,
    'p_value': p_value,
    'slope': slope,
    'intercept': intercept
  }