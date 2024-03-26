import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)  # Sigmoid activation function for binary classification

# Define the hyperparameters
input_size = 17  # Number of input features
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # Number of output classes (binary classification)

# Instantiate the model
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train.reshape(-1, 1))

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Convert test data to PyTorch tensors
X_test_tensor = torch.Tensor(X_test)

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy}')