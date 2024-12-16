# train/evaluate.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from config.config import DEVICE
from utils.metrics import calculate_metrics
from utils.utils import print_confusion_matrix

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the provided DataLoader and calculates metrics.

    Args:
        model: The trained model to evaluate.
        test_loader: DataLoader for the test set.

    Returns:
        A dictionary containing evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert lists to NumPy arrays for metric calculations
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate various metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"Specificity: {metrics['specificity']*100:.2f}%")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1']*100:.2f}%")
    print(f"Kappa: {metrics['kappa']:.4f}")
    
    # Print confusion matrix
    print_confusion_matrix(all_labels, all_preds)
    
    return metrics
