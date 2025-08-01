# src/model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from .config import LEARNING_RATE

def get_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=LEARNING_RATE)

def get_criterion():
    return nn.CrossEntropyLoss()

