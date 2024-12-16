import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from config.config import DATASET_PATHS, Workers

class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.classes = os.listdir(os.path.join(dataset_path, 'train'))
        self.image_paths = []
        self.labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(dataset_path, 'train', class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_train_transform():
    # Basic transform for dataset usage if needed
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_combined_train_loader(batch_size, transform):
    all_train_datasets = []
    for dataset_path in DATASET_PATHS:
        train_dataset = CustomDataset(dataset_path, transform=transform)
        all_train_datasets.append(train_dataset)
    combined_train_dataset = ConcatDataset(all_train_datasets)
    
    return DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=2,
        num_workers=Workers
    )
