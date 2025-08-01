import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import os

def load_model():
    # Dynamically get number of classes from dataset folder count
    data_dir = 'data/raw'
    classes = [d.name for d in os.scandir(data_dir) if d.is_dir()]
    num_classes = len(classes)
    print(f"✅ Number of classes detected: {num_classes}")

    # Load MobileNetV2 with pretrained weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Replace classifier with correct output dimension
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model_path = 'outputs/models/best_model.pth'
    if os.path.exists(model_path):
        print(f"📂 Loading model weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        print("⚠️ Model weights file not found, using fresh model.")

    model.eval()
    return model
