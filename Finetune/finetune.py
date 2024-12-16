# finetune.py

import argparse
import os
import warnings
import torch
import numpy as np
from data.dataloader import get_dataloaders
from train.train import train_model
from train.evaluate import evaluate_model
from config.config import (
    DEVICE, TEST_ONLY, DATASET_NAME, CHECKPOINT_PATH, BEST_MODEL_PATH
)
import warnings
import traceback

# Suppress specific warnings for cleaner output
warnings.filterwarnings(
    "ignore",
    message="You are using torch.load with weights_only=False",
    category=FutureWarning
)

def main(args):
    """
    Main function to handle training and testing based on command-line arguments.

    Args:
        args: Parsed command-line arguments.
    """
    if args.test_only:
        if not args.dataset:
            print("Error: Must specify dataset for test-only mode")
            exit(1)
        # Perform test-only evaluation
        test_metrics = train_model(dataset_name=args.dataset)
    else:
        print("Starting the training and evaluation process...")
        print(f"Using device: {DEVICE}")
        
        # Begin training
        print("\n=== Starting Combined Training ===")
        try:
            model, train_metrics = train_model()  # Training mode
            print("\nTraining completed successfully!")
            print("Training metrics:", train_metrics)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            exit(1)
        
        # Initialize results dictionary
        dataset_results = {}
        
        # Iterate through each dataset for individual testing
        print("\n=== Starting Individual Dataset Testing ===")
        for dataset in sorted(os.listdir('datasets')):
            dataset_path = os.path.join('datasets', dataset)
            if dataset != 'Combined' and os.path.isdir(dataset_path):
                print(f"\nTesting on {dataset} dataset:")
                try:
                    if os.path.exists(os.path.join(dataset_path, 'test')):
                        _, metrics = train_model(dataset_name=dataset)
                        dataset_results[dataset] = metrics
                        print(f"Results for {dataset}:")
                        for metric_name, value in metrics.items():
                            print(f"{metric_name}: {value:.4f}")
                    else:
                        print(f"Skipping {dataset} - no test set found")
                except Exception as e:
                    print(f"Error testing on {dataset}: {str(e)}")
                    traceback.print_exc()
                    continue
        
        # Display summary of all results
        print("\n=== Final Results Summary ===")
        print("\nPer-Dataset Results:")
        print("-" * 80)
        print(f"{'Dataset':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Specificity':>10} {'AUC':>10} {'F1-Score':>10}")
        print("-" * 80)
                
        for dataset, metrics in dataset_results.items():
            print(f"{dataset:<20} "
                f"{metrics['accuracy']*100:>10.2f} "
                f"{metrics['precision']*100:>10.2f} "
                f"{metrics['recall']*100:>10.2f} "
                f"{metrics['specificity']*100:>10.2f} "
                f"{metrics['auc']*100:>10.2f} "
                f"{metrics['f1']*100:>10.2f}")
                
        print("-" * 80)
        # Calculate and display average metrics
        if dataset_results:
            avg_metrics = {
                metric: np.mean([results[metric] for results in dataset_results.values()])
                for metric in ['accuracy', 'f1', 'auc', 'kappa']
            }
            print("\nAverage Metrics Across All Datasets:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value*100:.2f}%")
        
        print("\nProcess completed!")

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Train or test the model')
    parser.add_argument('--test-only', type=bool, default=False, help='Run in test-only mode')
    parser.add_argument('--model-path', type=str, default='checkpoints_large/best_finetuned_model.pth', help='Path to model weights')
    parser.add_argument('--dataset', type=str, default='IDRiD', help='Dataset to test on')
    args = parser.parse_args()
    
    main(args)
