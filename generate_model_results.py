import torch
import numpy as np
from torch.utils.data import DataLoader
from model import load_model, device
from AnimalDataset import AnimalDataset
from plot_accuracy_trends import get_shannon_entropy

def generate_model_results(model, dataloader, df_lookup, idx_to_species):
    """
    Generate model results for entropy analysis.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for test data
        df_lookup: AnimalDataset or DataFrame with test data info for entropy calculation
        idx_to_species: Dictionary mapping class indices to species names
    
    Returns:
        List of tuples: (entropy, correct) where:
        - entropy: Shannon entropy of volunteer classifications for this subject
        - correct: 1 if model prediction was correct, 0 if incorrect
    """
    model.eval()
    entropy_results = []
    
    # Handle different input types for df_lookup
    if hasattr(df_lookup, 'df'):
        dataframe = df_lookup.df
    elif hasattr(df_lookup, 'iterrows'):
        dataframe = df_lookup
    else:
        raise ValueError(f"df_lookup must be a pandas DataFrame or an AnimalDataset with a .df attribute")
    
    with torch.no_grad():
        batch_idx = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Convert to CPU and numpy
            predicted_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Process each sample in the batch
            for i in range(len(predicted_np)):
                # Calculate the actual index in the original dataset
                actual_idx = batch_idx * dataloader.batch_size + i
                
                if actual_idx < len(dataframe):
                    # Get the corresponding row from the dataframe
                    row = dataframe.iloc[actual_idx]
                    
                    # Calculate Shannon entropy for this subject
                    entropy_val = get_shannon_entropy(row)
                    
                    # Check if prediction is correct
                    is_correct = 1 if predicted_np[i] == labels_np[i] else 0
                    
                    # Store as tuple (entropy, correct)
                    entropy_results.append((entropy_val, is_correct))
                    
            batch_idx += 1
    
    return entropy_results