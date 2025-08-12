import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
  
def print_heatmap_normalized(confusion_matrix, idx_to_species, heatmap_path):
  labels = [idx_to_species[i] for i in range(len(idx_to_species))]
  
  cm_normalized = confusion_matrix.float().clone()
  for i in range(len(confusion_matrix)):
    row_sum = confusion_matrix[i].sum()
    if row_sum > 0: cm_normalized[i] = confusion_matrix[i] / row_sum
  
  plt.figure(figsize=(20, 15))
  sns.heatmap(cm_normalized.numpy(), 
              annot=np.array([[f'{cm_normalized[i,j]:.2f}\n({int(confusion_matrix[i,j])})'
                              for j in range(len(confusion_matrix))]
                            for i in range(len(confusion_matrix))]),
              fmt='', 
              cmap='Blues', 
              xticklabels=labels, 
              yticklabels=labels,
              vmin=0,
              vmax=1)
  
  plt.xticks(rotation=30, ha='right')
  plt.yticks(rotation=0)
  
  plt.xlabel('Predicted', labelpad=20)
  plt.ylabel('True', labelpad=20)
  plt.title('Confusion Matrix\n(proportion above, count below)', pad=20)
  
  plt.tight_layout()
  
  plt.savefig(heatmap_path, 
              bbox_inches='tight',
              dpi=300,
              pad_inches=0.5)
  
  plt.show()