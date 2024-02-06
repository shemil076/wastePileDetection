# from model_modification import modify_model
from utils.model_trainer import train_model
from utils.data_loader import load_data
from utils import evaluate_model
from ultralytics import YOLO

def main():
    # Load and modify the model
    model = YOLO('../models/best.pt')
    print(model)

    # Load data
    train_data, val_data, test_data = load_data('../data/rawData')

    # Train the model
    trained_model = train_model(model, train_data, val_data)

    # Evaluate the model
    evaluate_model(trained_model, test_data)

if __name__ == "__main__":
    main()
