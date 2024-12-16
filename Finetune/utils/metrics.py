# utils/metrics.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
)

def calculate_metrics(all_labels, all_preds, all_probs):
    """
    Calculates various evaluation metrics based on predictions and true labels.

    Args:
        all_labels: Ground truth labels.
        all_preds: Predicted labels.
        all_probs: Predicted probabilities.

    Returns:
        A dictionary containing calculated metrics.
    """
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(all_labels, all_preds)
    }
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError as e:
        print(f"Warning: Could not calculate AUC - {str(e)}")
        metrics['auc'] = 0.0
    
    # Calculate macro-averaged specificity for multi-class
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    specificities = []
    
    for i in range(n_classes):
        # True negatives are all examples not in class i correctly predicted as not class i
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        # False positives are examples incorrectly predicted as class i
        fp = np.sum(np.delete(cm[:, i], i))
        
        # Calculate specificity for this class
        if tn + fp == 0:
            spec = 0.0
        else:
            spec = tn / (tn + fp)
        specificities.append(spec)
    
    # Use macro-averaged specificity
    metrics['specificity'] = np.mean(specificities)
    
    return metrics
