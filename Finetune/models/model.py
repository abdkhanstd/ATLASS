# models/model.py

import os
import torch
import torch.nn as nn
import timm
from config.config import BEST_MODEL_PATH, DEVICE

def create_vit_model(num_classes, verbose=False):
    """
    Creates a ViT-Large model with a customized classification head.
    
    Args:
        num_classes: Number of output classes for classification.
        verbose: If True, prints detailed information during model creation.

    Returns:
        The ViT-Large model ready for training or evaluation.
    """
    if verbose:
        print(f"\nCreating ViT-Large model for {num_classes} classes...")
    
    # Initialize the ViT-Large model with pretrained weights
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    
    # Retrieve the input feature size of the original classification head
    in_features = model.head.in_features
    
    # Define a new classification head with multiple layers and dropout
    model.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 2048),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.LayerNorm(2048),
        nn.Linear(2048, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.LayerNorm(1024),
        nn.Linear(1024, num_classes)
    )
    
    # Initialize weights of the new classification head
    for module in model.head:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    # Freeze all pretrained weights except the classification head
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    
    # Display parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Load self-supervised pretrained weights if available
    if os.path.exists(BEST_MODEL_PATH):
        if verbose:
            print(f"Loading self-supervised weights from {BEST_MODEL_PATH}")
            
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        
        # Extract state dictionary from the checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Filter out head-related weights and ensure compatibility
        new_state_dict = {}
        for k, v in state_dict.items():
            if not any(skip in k for skip in ['head', 'fc', 'classifier']):
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
                    if verbose:
                        print(f"Loaded {k} with shape {v.shape}")
        
        # Load the filtered weights into the model
        missing = model.load_state_dict(new_state_dict, strict=False)
        if verbose:
            print(f"Loaded encoder weights. Missing keys: {len(missing.missing_keys)}")
    
    return model.to(DEVICE)
