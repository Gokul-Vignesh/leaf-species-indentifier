# src/data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from .config import RAW_DIR, BATCH_SIZE, IMG_SIZE

def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(RAW_DIR, transform=transform)
    n = len(dataset)
    val_size = int(0.2 * n)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, dataset.classes

