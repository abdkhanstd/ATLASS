import torch
import torch.optim as optim
import traceback
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)

import torchvision.transforms as transforms

from config.config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINT_PATH,
    BEST_MODEL_PATH, Patience, Workers, ACCUMULATION_STEPS,
    SAVE_SAMPLES, SAMPLE_INTERVAL, SAMPLES_DIR,
    VIT_WEIGHTS_PATH, DECODER_WEIGHTS_PATH
)
from data.dataset import get_combined_train_loader
from models.vit_models import (
    ViTPatch16LargeFullReconstruction,
    VesselMaskGenerator,
    RestorationLoss
)
from utils.utils import (
    save_vessel_samples,
    generate_random_crops,
    calculate_psnr,
    save_vit_weights,
    save_decoder_weights
)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def train_reconstruction():
    print("\nInitializing Self Supervised pretraining...")
    model = ViTPatch16LargeFullReconstruction(use_gradient_checkpointing=True)
    model = model.to(DEVICE)
    print("Main model initialized")
    
    vessel_generator = VesselMaskGenerator(
        model_path='vessel_best_model.pth',
        threshold=0.5,
        remove_small_objects=True,
        min_object_size=50
    )
    print("Vessel mask generator initialized")
    
    # Test vessel mask generation
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        test_mask = vessel_generator.generate_vessel_mask(test_input)
    
    for param in model.parameters():
        param.requires_grad = True
    
    scaler = torch.amp.GradScaler()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = get_combined_train_loader(BATCH_SIZE, transform)
    print(f"Created dataloader with {len(train_loader)} batches")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    criterion = RestorationLoss().to(DEVICE)
    best_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch_idx, (inputs, _) in enumerate(pbar):
                try:
                    inputs = inputs.to(DEVICE)
                    vessel_masks = vessel_generator.generate_vessel_mask(inputs)
                    random_masks = generate_random_crops(
                        batch_size=inputs.size(0),
                        height=224,
                        width=224,
                        device=DEVICE
                    )
                    
                    combined_masks = torch.maximum(
                        vessel_masks.squeeze(1),
                        random_masks * 0.8
                    ).unsqueeze(1)
                    
                    masked_input = inputs * (~combined_masks.expand(-1, 3, -1, -1).bool())
                    
                    with torch.amp.autocast(device_type='cuda'):
                        reconstructed = model(masked_input)
                        reconstructed = torch.sigmoid(reconstructed)
                        loss, loss_components = criterion(reconstructed, inputs, combined_masks)
                        loss = loss / ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    running_loss += loss.item() * ACCUMULATION_STEPS
                    
                    if SAVE_SAMPLES and batch_idx % SAMPLE_INTERVAL == 0:
                        save_vessel_samples(
                            inputs.clone(),
                            reconstructed.detach(),
                            combined_masks.clone(),
                            epoch,
                            batch_idx
                        )
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}",
                        **loss_components
                    })
                
                except torch.cuda.OutOfMemoryError:
                    print("\nOOM error - cleaning up and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"\nError in training step: {str(e)}")
                    traceback.print_exc()
                    continue
        
        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f"New best loss: {best_loss:.4f} - Saving model...")
            save_vit_weights(model.vit, VIT_WEIGHTS_PATH)
            save_decoder_weights(model, DECODER_WEIGHTS_PATH)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, BEST_MODEL_PATH)
            print("Model saved successfully")
        else:
            patience_counter += 1
        
        if patience_counter >= Patience:
            print(f"\nNo improvement for {Patience} epochs. Stopping training.")
            break
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    print("\nTraining completed!")
    return model



if __name__ == "__main__":
    model = train_reconstruction()
    print("Training completed successfully!")
