from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

import argparse
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd

num_features = 12
test_size = 0.25

def read_data(df):
    gameids = df['gameid'].tolist()
    # gamenum = df['momentum'].tolist()
    df = df.drop(columns=['gameid'])
    X = df.drop(columns=['result'])  # Assuming 'result' is the target column
    y = df['result']

    # Initialize StratifiedShuffleSplit with desired parameters
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

    # Split the data into train and combined test/dev sets
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # # Further split the combined test/dev set into development and test sets
    # sss_dev_test = StratifiedShuffleSplit(n_splits=1, test_size=0.66)
    # for dev_index, test_index in sss_dev_test.split(X_test_dev, y_test_dev):
    #     X_dev, X_test = X_test_dev.iloc[dev_index], X_test_dev.iloc[test_index]
    #     y_dev, y_test = y_test_dev.iloc[dev_index], y_test_dev.iloc[test_index]

    return gameids, X_train, y_train, X_test, y_test

def predict(model, X):
    """Return the predictions using weight vector w on inputs X.

    Args:
        - w: Vector of size (D,)
        - X: Matrix of size (M, D)
    Returns:
        - Predicted classes as a numpy vector of size (M,). Each entry should be either -1 or +1.
    """
    prob = model.predict(X)
    print(prob.shape)
    return (prob > 0.5).astype(int)

def evaluate(model, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    y_preds = predict(model, X)
    print("Predictions:")
    print(y.shape)

    acc = np.mean(y_preds.flatten() == y)

    total = len(y)
    correct = int(acc*total)
    print(str(correct) + " of " + str(total) +" correct")

    print('    {} Accuracy: {}'.format(name, acc))
    return acc

# Define the DNN model
model = Sequential([
    # Input layer
    Dense(64, activation='relu', input_shape=(num_features,)),
    
    # Hidden layers
    Dense(128, activation='relu'),
    Dropout(0.2),  # Optional dropout layer to prevent overfitting
    BatchNormalization(),  # Optional batch normalization layer
    
    Dense(64, activation='relu'),
    Dropout(0.2),  # Optional dropout layer to prevent overfitting
    BatchNormalization(),  # Optional batch normalization layer
    
    # Output layer
    Dense(1, activation='sigmoid')  # 3 output neurons for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit model
input_csv = "../data/output/LCK_LogReg_Dataset_No_F4.csv"
input_df = pd.read_csv(input_csv, index_col=0)
gameids, X_train, y_train, X_test, y_test = read_data(input_df)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
train_acc = history.history['val_accuracy'][-1]
dev_acc = history.history['accuracy'][-1]

# Evaluate model
print('    Train Accuracy: {}'.format(train_acc))
print('    Dev Accuracy: {}'.format(dev_acc))
test_acc = evaluate(model, X_test, y_test, 'Test')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # Define the neural network architecture
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer
#         self.relu = nn.ReLU()  # ReLU activation function
#         self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return torch.sigmoid(x)  # Sigmoid activation function for binary classification

# # Define the hyperparameters
# input_size = 17  # Number of input features
# hidden_size = 64  # Number of neurons in the hidden layer
# output_size = 1  # Number of output classes (binary classification)

# # Instantiate the model
# model = NeuralNetwork(input_size, hidden_size, output_size)

# # Define the loss function and optimizer
# criterion = nn.BCELoss()  # Binary cross-entropy loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# # Convert data to PyTorch tensors
# X_train_tensor = torch.Tensor(X_train)
# y_train_tensor = torch.Tensor(y_train.reshape(-1, 1))

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Print loss for every epoch
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# # Convert test data to PyTorch tensors
# X_test_tensor = torch.Tensor(X_test)

# # Evaluate the model
# with torch.no_grad():
#     outputs = model(X_test_tensor)
#     predicted = (outputs >= 0.5).float()
#     accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
#     print(f'Test Accuracy: {accuracy}')