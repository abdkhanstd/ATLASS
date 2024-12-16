# data/utils.py

import os
from .dataset import FineTuneDataset

def get_class_mapping_indices(dataset_name):
    """
    Maps dataset-specific classes to JSIEC indices using FineTuneDataset mappings.

    Args:
        dataset_name: Name of the dataset to map.

    Returns:
        A dictionary mapping original dataset class indices to JSIEC class indices.
    """
    # Initialize temporary dataset to access class mappings
    dataset = FineTuneDataset(
        dataset_path=os.path.join('datasets', dataset_name),
        split='train'
    )
    
    # Retrieve original dataset classes in sorted order
    dataset_classes = sorted(os.listdir(os.path.join('datasets', dataset_name, 'train')))
    
    # Initialize mapping dictionary
    dataset_to_jsiec_idx = {}
    
    # Retrieve the specific mapping for the dataset
    mapping = dataset.class_mappings.get(dataset_name, {})
    
    # Iterate through each original class and map to JSIEC index
    for idx, orig_class in enumerate(dataset_classes):
        if orig_class in mapping:
            jsiec_class = mapping[orig_class]
            jsiec_idx = dataset.class_to_idx.get(jsiec_class, None)
            if jsiec_idx is not None:
                dataset_to_jsiec_idx[idx] = jsiec_idx
    
    # Display the class mapping for verification
    print("\nClass mapping for", dataset_name)
    print("Original idx -> JSIEC idx")
    for orig_idx, jsiec_idx in dataset_to_jsiec_idx.items():
        print(f"{orig_idx} -> {jsiec_idx}")
        
    return dataset_to_jsiec_idx
