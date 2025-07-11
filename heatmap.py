import seaborn as sns
import matplotlib.pyplot as plt

def print_heatmap(confusion_matrix, idx_to_species):

  labels = [idx_to_species[i] for i in range(len(idx_to_species))]

  plt.figure(figsize=(10, 8))
  sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)  
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()
  plt.savefig("confusion_matrix_pretrained_1.png")