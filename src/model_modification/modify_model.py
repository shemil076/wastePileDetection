import torch
import torch.nn as nn
from ultralytics import YOLO

def modify_model():
    # Load the YOLOv8 model
    model_path = '../model/best.pt'
    model = torch.load(model_path)
    model.eval()

    # Apply modifications
    # For example, let's add a new linear layer at the end of the model
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)  # Assuming we want to classify into 2 classes

    return model
