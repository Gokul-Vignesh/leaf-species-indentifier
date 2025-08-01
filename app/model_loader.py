# app/model_loader.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Load your class labels (e.g., Abies concolor, Acer rubrum, etc.)
CLASS_NAMES = [
    "Abies concolor", "Acer rubrum", "Quercus alba", "Pinus strobus"
    # Add more from your dataset...
]

# Define the transform (should match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model(model_path='outputs/models/best_model.pth'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict function
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

