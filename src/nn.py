# Description: Neural Network model for predicting match outcomes

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Import other necessary libraries
import argparse
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd

val_size = 0.2 # validation size (of train + val set)
test_size = 0.25 # test size (of total data)
num_features = 12 # features

total_iters = 10 # total number of iterations (to run the model) w/ same hyperparameters to get average accuracy
epoch_num_hyp = 20 # number of epochs to train the model on 
batch_size_hyp = 32 # num_samples per batch for training the model
batch_norm = True # batch normalization which is a technique to improve the training of deep learning models

loss_hyp = 'binary_crossentropy' # binary_crossentropy loss function

optimizer = 'AdamW' # Adam or AdamW
init_learning_rate = 0.001 # initial learning rate for the optimizer
weight_decay_hyp = 0.01 # weight decay for AdamW optimizer

dropout_rate = 0.4 # dropout rate to prevent overfitting
patience_epochs = 10 # early stopping patience aka number of epochs with no improvement after which training will be stopped

class HyperParams:
    def __init__(self, val_size, test_size, num_features, total_iters, num_epochs, batch_size, batch_norm, loss_function, optimizer, init_learning_rate, weight_decay, dropout_rate, patience_epochs):
        self.val_size = val_size
        self.test_size = test_size
        self.num_features = num_features

        self.total_iters = total_iters
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.batch_norm = batch_norm

        self.loss_function = loss_function

        self.optimizer_name = optimizer
        self.optimizer = None
        self.init_learning_rate = init_learning_rate
        self.weight_decay = weight_decay
        
        # Learning rate schedule for AdamW Optimizer
        self.decay_steps = 10000
        self.decay_rate = 0.95   

        self.dropout_rate = dropout_rate
        self.patience_epochs = patience_epochs

    # getters
    def get_init_learning_rate(self):
        return self.init_learning_rate
    
    def get_weight_decay(self):
        return self.weight_decay
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_epochs(self):
        return self.epochs
    
    def get_dropout_rate(self):
        return self.dropout_rate
    
    def get_batch_norm(self):
        return self.batch_norm
    
    def get_optimizer_name(self):  
        return self.optimizer_name
    
    def get_optimizer(self):
        if self.optimizer_name == 'Adam':
            self.optimizer = Adam(learning_rate=self.init_learning_rate)
        elif self.optimizer_name == 'AdamW':
            # Define learning rate schedule && AdamW optimizer
            learning_rate_fn = ExponentialDecay(self.init_learning_rate, self.decay_steps, self.decay_rate, staircase=True )
            optimizer = Adam(learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay=self.weight_decay) # 
            self.optimizer = optimizer
        return self.optimizer
    
    def get_loss_function(self): 
        return self.loss_function
    
    def get_num_features(self):
        return self.num_features
    
    def set_num_features(self, num_features):
        self.num_features = num_features
    
    def get_val_size(self):
        return self.val_size
    
    def get_test_size(self):
        return self.test_size
    
    # Learning rate schedule for AdamW Optimizer
    def get_decay_steps(self):
        return self.decay_steps
    
    def get_decay_rate(self):
        return self.decay_rate
    
    def get_patience_epochs(self):
        return self.patience_epochs
    
    def get_total_iters(self):
        return self.total_iters    
    
    # Print hyperparameters
    def print_hyperparameters(self):
        print("HYPERPARAMETERS")
        print(f" ... {1 - ((1 - self.test_size) * self.val_size)}-{(1 - self.test_size) * self.val_size}-{self.test_size} train-val-test split on {self.num_features} features")
        print(f" ... Learning Rate: {self.init_learning_rate} | Weight Decay: {self.weight_decay}")
        print(f" ... Batch Size: {self.batch_size} | Epochs: {self.epochs} | Batch Normalization: {self.batch_norm}")
        print(f" ... Optimizer: {self.optimizer_name} | Loss Function: {self.loss_function} | Dropout Rate: {self.dropout_rate}")
    
    def return_hyperparameters(self):
        output_msg = "HYPERPARAMETERS" + "\n" + f" ... {1 - ((1 - self.test_size) * self.val_size)}-{(1 - self.test_size) * self.val_size}-{self.test_size} train-val-test split on {self.num_features} features" + "\n" + f" ... Learning Rate: {self.init_learning_rate} | Weight Decay: {self.weight_decay}" + "\n" + f" ... Batch Size: {self.batch_size} | Epochs: {self.epochs} | Batch Normalization: {self.batch_norm}" + "\n" + f" ... Optimizer: {self.optimizer_name} | Loss Function: {self.loss_function} | Dropout Rate: {self.dropout_rate}"
        return output_msg
        
class NeuralNetwork:
    # Define the DNN model
    def __init__(self, num_features, dropout_rate, batch_norm=True):
        self.input_shape = (num_features,)
        self.input_layer = Input(shape=self.input_shape)
        self.hidden_layer1 = Dense(64, activation='relu')(self.input_layer)
        self.hidden_layer2 = Dense(128, activation='relu')(self.hidden_layer1)
        self.dropout_layer1 = Dropout(dropout_rate)(self.hidden_layer2)

        if batch_norm:
            self.batchnorm_layer1 = BatchNormalization()(self.dropout_layer1)
            self.hidden_layer3 = Dense(64, activation='relu')(self.batchnorm_layer1)
        else:
            self.hidden_layer3 = Dense(64, activation='relu')(self.dropout_layer1)

        self.dropout_layer2 = Dropout(dropout_rate)(self.hidden_layer3)

        if batch_norm:
            self.batchnorm_layer2 = BatchNormalization()(self.dropout_layer2)
            self.output_layer = Dense(1, activation='sigmoid')(self.batchnorm_layer2)
        else: 
            self.output_layer = Dense(1, activation='sigmoid')(self.dropout_layer2)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)

    def get_model(self):
        return self.model
    
    def compile_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def print_model_summary(self):
        self.model.summary()

# Data Split ... 
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

# Define Hyperparameters
# val_size | test_size | num_features
# total_iters | num_epochs | batch_size | batch_norm | 
# loss_function
# optimizer | init_learning_rate | weight_decay
# dropout_rate | patience_epochs
params = HyperParams(val_size, 
                     test_size, 
                     num_features, 
                     total_iters, 
                     epoch_num_hyp, 
                     batch_size_hyp, 
                     batch_norm, 
                     loss_hyp, 
                     optimizer, 
                     init_learning_rate, 
                     weight_decay_hyp,
                     dropout_rate,
                     patience_epochs)

# Fit match_prediction_model
input_csv = "../data/output/LCK_LogReg_Dataset_No_F4.csv"
input_df = pd.read_csv(input_csv, index_col=0)
gameids, X_train, y_train, X_test, y_test = read_data(input_df)
params.set_num_features(X_train.shape[1])

DNN = NeuralNetwork(params.get_num_features(), params.get_dropout_rate(), params.get_batch_norm())
match_prediction_model = DNN.get_model()
DNN.print_model_summary()

DNN.compile_model(params.get_optimizer(), params.get_loss_function())

print ("Training Data Shape: ", X_train.shape)
print ("Testing Data Shape: ", X_test.shape)

min_train_acc_set = [1, 1, 1]
min_dev_acc_set = [1, 1, 1]
min_test_acc_set = [1, 1, 1]
max_train_acc_set = [0, 0, 0]
max_dev_acc_set = [0, 0, 0]
max_test_acc_set = [0, 0, 0]

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for num_iters in range(total_iters): # ranges from 0 to total_iters-1
    # reset all values
    history = None
    train_acc = 0
    dev_acc = 0
    test_acc = 0

    msg = "ITERATION " + str(num_iters) + "\n"
    print(msg)

    history = match_prediction_model.fit(X_train, y_train, epochs=params.get_epochs(), batch_size=params.get_batch_size(), validation_split=params.get_val_size(), callbacks=[early_stopping]) # validation size
    train_acc = history.history['accuracy'][-1]
    dev_acc = history.history['val_accuracy'][-1]

    # Evaluate match_prediction_model
    print('    Train Accuracy: {}'.format(train_acc))
    print('    Dev Accuracy: {}'.format(dev_acc))
    test_acc = evaluate(match_prediction_model, X_test, y_test, 'Test')

    params.print_hyperparameters() # print hyperparameters

    if train_acc < min_train_acc_set[0]:
        min_train_acc_set = [train_acc, dev_acc, test_acc]
    if dev_acc < min_dev_acc_set[1]:
        min_dev_acc_set = [train_acc, dev_acc, test_acc]
    if test_acc < min_test_acc_set[2]:
        min_test_acc_set = [train_acc, dev_acc, test_acc]
    if train_acc > max_train_acc_set[0]:
        max_train_acc_set = [train_acc, dev_acc, test_acc]
    if dev_acc > max_dev_acc_set[1]:
        max_dev_acc_set = [train_acc, dev_acc, test_acc]
    if test_acc > max_test_acc_set[2]:
        max_test_acc_set = [train_acc, dev_acc, test_acc]

print("MINIMUM ACCURACIES")
print("    [TRAIN] *TRAIN*:", min_train_acc_set[0], "dev:", min_train_acc_set[1], "test:", min_train_acc_set[2])
print("    [DEV] train:", min_dev_acc_set[0], "*DEV*:", min_dev_acc_set[1], "test:", min_dev_acc_set[2])
print("    [TEST] train:", min_test_acc_set[0], "dev:", min_test_acc_set[1], "*TEST*:", min_test_acc_set[2])

print("MAXIMUM ACCURACIES")
print("    [TRAIN] *TRAIN*:", max_train_acc_set[0], "dev:", max_train_acc_set[1], "test:", max_train_acc_set[2])
print("    [DEV] train:", max_dev_acc_set[0], "*DEV*:", max_dev_acc_set[1], "test:", max_dev_acc_set[2])
print("    [TEST] train:", max_test_acc_set[0], "dev:", max_test_acc_set[1], "*TEST*:", max_test_acc_set[2])
