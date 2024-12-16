# utils/utils.py

import os
import warnings
from sklearn.metrics import confusion_matrix

def suppress_warnings():
    """
    Suppresses specific warnings to keep the output clean.
    """
    warnings.filterwarnings(
        "ignore",
        message="You are using torch.load with weights_only=False",
        category=FutureWarning
    )

def print_confusion_matrix(all_labels, all_preds):
    """
    Prints the confusion matrix for the predictions.

    Args:
        all_labels: Ground truth labels.
        all_preds: Predicted labels.
    """
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
