# config/config.py

import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
EPOCHS = 25
PATIENCE = 5
WEIGHT_DECAY = 0.01

# Learning rate scheduler parameters
LR_REDUCE_FACTOR = 0.5
LR_PATIENCE = 5
LR_MIN = 1e-7

# Dataset and checkpoint configurations
DATASET_NAME = 'IDRiD'
CHECKPOINT_PATH = './checkpoints_large'
SAVE_DIR = os.path.join(CHECKPOINT_PATH, DATASET_NAME)
WEIGHTS_PATH = os.path.join(CHECKPOINT_PATH, 'vit_weights.pth')
BEST_MODEL_PATH = WEIGHTS_PATH

# Create necessary directories
os.makedirs(SAVE_DIR, exist_ok=True)

TEST_ONLY=True