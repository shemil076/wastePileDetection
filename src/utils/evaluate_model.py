from sklearn.metrics import accuracy_score

# def evaluate_model(model, test_data):
#     # Separate features and target
#     X_test, y_test = test_data

#     # Make predictions
#     predictions = model.predict(X_test)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, predictions)

#     # Print out metrics
#     print(f'Accuracy: {accuracy * 100:.2f}%')

#     return accuracy


from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model's accuracy.

    Parameters:
    model: The trained model object.
    X_test: Test features.
    y_test: True labels for the test data.

    Returns:
    float: The accuracy of the model on the test data.
    """
    # Predicting the labels for the test set
    y_pred = model.predict(X_test)

    # Calculating the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

