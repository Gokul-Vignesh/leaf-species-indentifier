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
from PIL import Image
from torchvision import transforms
import torch

def predict_image(image_path, model):
    """Predict the class of a given image using the trained model."""
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = output.max(1)

    class_names_path = 'outputs/class_names.txt'
    if not os.path.exists(class_names_path):
        raise FileNotFoundError("class_names.txt not found in outputs/")

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    return class_names[predicted_class.item()]
