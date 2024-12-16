# train/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config.config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, 
    CHECKPOINT_PATH, BEST_MODEL_PATH, WEIGHT_DECAY
)
from models.model import create_vit_model
from data.dataloader import get_dataloaders
from utils.metrics import calculate_metrics
from train.evaluate import evaluate_model


def train_model(dataset_name=None):
    """
    Trains the model on combined datasets or evaluates on a specific dataset.

    Args:
        dataset_name: If provided, runs in test-only mode on the specified dataset.

    Returns:
        Trained model and corresponding metrics.
    """
    if dataset_name:  # Testing mode
        # Retrieve DataLoader and label mapping for the specific dataset
        test_loader, label_mapping = get_dataloaders(None, dataset_name)
        
        # Determine the number of classes from the mapping
        num_classes = len(set(label_mapping.values()))
        print(f"Testing {dataset_name} with {num_classes} classes")
        
        # Initialize the model with the correct number of classes
        model = create_vit_model(num_classes=num_classes)
        
        # Load the trained model weights
        trained_state = torch.load(os.path.join(CHECKPOINT_PATH, 'best_finetuned_model.pth'), 
                                   map_location=DEVICE, weights_only=True)
        
        # Initialize a new state dictionary for mapping weights
        new_state_dict = {}
        for k, v in trained_state.items():
            if 'head.10.weight' in k:  # Final classification layer weights
                # Create a new weight matrix with the correct shape
                new_weights = torch.zeros((num_classes, v.size(1)), device=v.device)
                for orig_idx, new_idx in label_mapping.items():
                    if orig_idx < v.size(0):  # Ensure index is within bounds
                        new_weights[new_idx] = v[orig_idx]
                new_state_dict[k] = new_weights
            elif 'head.10.bias' in k:  # Final classification layer biases
                # Create a new bias vector with the correct shape
                new_bias = torch.zeros(num_classes, device=v.device)
                for orig_idx, new_idx in label_mapping.items():
                    if orig_idx < v.size(0):  # Ensure index is within bounds
                        new_bias[new_idx] = v[orig_idx]
                new_state_dict[k] = new_bias
            elif 'head' in k:
                # Copy all other head layer weights directly
                new_state_dict[k] = v
            else:
                # Copy all non-head weights directly
                new_state_dict[k] = v
        
        # Load the mapped weights into the model
        missing = model.load_state_dict(new_state_dict, strict=False)
        if missing.missing_keys:
            print("Missing keys:", missing.missing_keys)
        if missing.unexpected_keys:
            print("Unexpected keys:", missing.unexpected_keys)
        
        model = model.to(DEVICE)
        
        # Evaluate the model
        metrics = evaluate_model(model, test_loader)
        print(f"\nTest Metrics for {dataset_name}:", metrics)
        return model, metrics
    
    else:  # Training mode
        # Retrieve DataLoaders for training and validation
        train_loader, val_loader, num_classes = get_dataloaders(None)
        print(f"Training model with {num_classes} classes")
        
        # Initialize the model
        model = create_vit_model(num_classes=num_classes)
        
        # Define optimizer with different learning rates and weight decay
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'head' not in n], 
             'lr': LEARNING_RATE * 0.1,
             'weight_decay': WEIGHT_DECAY},
            {'params': model.head.parameters(), 
             'lr': LEARNING_RATE,
             'weight_decay': WEIGHT_DECAY * 0.1}
        ])
        
        # Define loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Initialize mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()
        
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        best_val_acc = 0
        patience_counter = 0
        
        print("\nStarting training...")
        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Acc':^8}")
        print("-" * 50)
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Iterate over training DataLoader with progress bar
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{running_loss/len(train_loader):.4f}",
                        'acc': f"{100.*correct/total:.2f}%",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
            
            # Calculate average training loss and accuracy
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Perform validation
            val_metrics = evaluate_model(model, val_loader)
            val_acc = val_metrics['accuracy'] * 100
            
            # Display epoch results
            print(f"{epoch+1:^6} | {train_loss:^10.4f} | {train_acc:^9.2f} | {val_acc:^8.2f}")
            
            # Update learning rate scheduler
            scheduler.step()
            
            # Check for improvement and implement early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, 'best_finetuned_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("\nEarly stopping triggered!")
                    break
        
        print("\nTraining completed!")
        return model, val_metrics
