# data/dataloader.py

import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from .dataset import FineTuneDataset
from .utils import get_class_mapping_indices

def get_dataloaders(dataset_handler, specific_dataset=None):
    """
    Constructs DataLoaders for training, validation, or testing with appropriate transformations.
    
    Args:
        dataset_handler: Instance of MultiDatasetHandler (not used but kept for compatibility).
        specific_dataset: Name of the specific dataset for testing.

    Returns:
        For testing:
            - test_loader: DataLoader for the test set.
            - label_mapping: Dictionary mapping original class indices to new indices.
        For training:
            - train_loader: DataLoader for the training set.
            - val_loader: DataLoader for the validation set.
            - num_classes: Number of classes in the dataset.
    """
    # Define data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomAdjustSharpness(2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define transformations for evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if specific_dataset:  # Testing mode
        # Initialize dataset for testing with JSIEC mappings
        dataset_path = os.path.join('datasets', specific_dataset)
        test_dataset = FineTuneDataset(dataset_path, 'test', eval_transform)
        
        # Retrieve class mapping for the specific dataset
        class_mapping = get_class_mapping_indices(specific_dataset)
        
        # Extract unique JSIEC indices used in this dataset
        jsiec_indices = sorted(set(class_mapping.values()))
        
        # Create a mapping from JSIEC indices to consecutive indices
        jsiec_to_consecutive = {jsiec_idx: cons_idx for cons_idx, jsiec_idx in enumerate(jsiec_indices)}
        
        # Remap sample labels to consecutive indices
        remapped_samples = []
        for path, label in test_dataset.samples:
            if label in jsiec_to_consecutive:
                new_label = jsiec_to_consecutive[label]
                remapped_samples.append((path, new_label))
            else:
                print(f"Warning: Label {label} not found in mapping")
        
        if len(remapped_samples) != len(test_dataset.samples):
            print(f"Warning: Some samples were dropped due to mapping ({len(remapped_samples)} vs {len(test_dataset.samples)})")
        
        # Update dataset samples with remapped labels
        test_dataset.samples = remapped_samples
        
        # Create DataLoader for testing
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,  # Using global BATCH_SIZE from config
            shuffle=False,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True
        )
        
        return test_loader, jsiec_to_consecutive

    else:  # Training mode
        # Retrieve all dataset paths except 'Combined'
        dataset_paths = [os.path.join('datasets', d) for d in os.listdir('datasets') 
                        if os.path.isdir(os.path.join('datasets', d)) and d != 'Combined']
        
        train_datasets = []
        val_datasets = []
        
        for path in dataset_paths:
            try:
                # Initialize training and validation datasets
                train_dataset = FineTuneDataset(path, 'train', train_transform)
                val_dataset = FineTuneDataset(path, 'val', eval_transform)
                
                if len(train_dataset) > 0:
                    train_datasets.append(train_dataset)
                    
                if len(val_dataset) > 0:
                    val_datasets.append(val_dataset)
                    
            except Exception as e:
                print(f"Skipping {path}: {str(e)}")
                continue
        
        if not train_datasets or not val_datasets:
            raise ValueError("No valid datasets found")
        
        # Concatenate all training and validation datasets
        combined_train = ConcatDataset(train_datasets)
        combined_val = ConcatDataset(val_datasets)
        
        # Create DataLoaders for training and validation
        train_loader = DataLoader(
            combined_train,
            batch_size=64,  # Using global BATCH_SIZE from config
            shuffle=True,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            combined_val,
            batch_size=64,  # Using global BATCH_SIZE from config
            shuffle=False,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, 39  # 39 classes for JSIEC
