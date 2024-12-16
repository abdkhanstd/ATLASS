import os
import torch

# Configuration constants (preserving original hyperparameters)
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 300
CHECKPOINT_PATH = './checkpoints_large'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'best_restoration_model.pth')
Patience = 50
Workers = 10
ACCUMULATION_STEPS = 16

# Visualization settings
SAVE_SAMPLES = True
SAMPLE_INTERVAL = 20
SAMPLES_DIR = os.path.join(CHECKPOINT_PATH, 'samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIT_WEIGHTS_PATH = os.path.join(CHECKPOINT_PATH, 'vit_weights.pth')
DECODER_WEIGHTS_PATH = os.path.join(CHECKPOINT_PATH, 'decoder_weights.pth')

# Determine dataset paths
DATASET_PATHS = [
    os.path.join('datasets', d) 
    for d in os.listdir('datasets') 
    if os.path.isdir(os.path.join('datasets', d))
]
