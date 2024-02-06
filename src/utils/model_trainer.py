import torch

def train_model(model, train_data, val_data, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    best_model = None

    print(train_data)
    print(val_data)

    return model.train(data='/Users/pramudithakarunarathna/Documents/IIT Final Year/FYP/Implementations/wastePileDetection/data/dataset.yaml', epochs=10)  # Set model to training mode

    # for epoch in range(num_epochs):
    #     print("ran16")
    #     model.train(data='../data/dataset.yaml', epochs=10)  # Set model to training mode
    #     print("ran18")
    #     train_loss = 0.0

    #     # Iterate over the training data
    #     for inputs, labels in train_data:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()  # Zero the parameter gradients
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         train_loss += loss.item()
    #         loss.backward()  # Backward pass and optimize
    #         optimizer.step()

    #     avg_train_loss = train_loss / len(train_data)  # Average training loss

    #     model.eval()  # Set model to evaluate mode
    #     val_loss = 0.0
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, labels in val_data:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #             val_loss += loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     avg_val_loss = val_loss / len(val_data)  # Average validation loss
    #     val_accuracy = correct / total  # Validation accuracy

    #     # Save the best model
    #     if val_accuracy > best_val_accuracy:
    #         best_val_accuracy = val_accuracy
    #         best_model = model.state_dict()

    #     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, '
    #           f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Load the best model before returning
    # if best_model is not None:
    #     model.load_state_dict(best_model)

    # return model