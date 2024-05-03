import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention

# Load the dataset
input_csv = "../data/2023/2023_LCK_LogReg_Dataset_No_F4.csv"
matches_data = pd.read_csv(input_csv)

# Data preprocessing
# Normalize input features
# scaled_features = matches_data.drop(columns=['result', 'gameid'])
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(matches_data.drop(columns=['result', 'gameid']))
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_features = scaler.fit_transform(matches_data.drop(columns=['result', 'gameid']))

# Create input sequences and corresponding output labels
sequence_length = 10  # Adjust as needed
X, y = [], []
for i in range(sequence_length, len(matches_data)):
    X.append(scaled_features[i - sequence_length:i])
    y.append(matches_data.iloc[i]['result'])
X, y = np.array(X), np.array(y)

# number of samples in data
n_samples = X.shape[0]
print(f"Number of samples: {n_samples}")

# Split the dataset into training, validation, and test sets

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.30, random_state=42)

# Define the LSTM model architecture
model = Sequential([
    # LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    # Dense(1, activation='sigmoid') # Binary classification
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),  # Add another LSTM layer
    Dropout(0.2),
    LSTM(32),  # Add another LSTM layer
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# kf = KFold(n_splits=5)
# for train_index, val_index in kf.split(X):
#     X_train, X_val = X[train_index], X[val_index]
#     y_train, y_val = y[train_index], y[val_index]
# Train the model

# epochs = num passes through entire dataset larger = better but more time
# batch_size = num samples per batch ... larger = better but more expensive
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=70, batch_size=64)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
