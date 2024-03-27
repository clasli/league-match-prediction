"""Code for HW2 Problem 1: Logistic Regression with regularization on MNIST."""
import argparse
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split

import pandas as pd

OPTS = None

NEGATIVE_CLASS = '5'  # Represented as y=-1
POSITIVE_CLASS = '8'  # Represented as y=+1
IMAGE_SHAPE = (28, 28)  # Size of MNIST images

def predict(w, X):
    """Return the predictions using weight vector w on inputs X.

    Args:
        - w: Vector of size (D,)
        - X: Matrix of size (M, D)
    Returns:
        - Predicted classes as a numpy vector of size (M,). Each entry should be either -1 or +1.
    """
    signs = np.sign(X.dot(w.T))
    # print(w)
    return np.where(signs == -1, 0, signs)

def train(X_train, y_train, lr=1e-1, num_iters=5000, l2_reg=0.0):
    """Train linear regression using gradient descent.

    Args:
        - X_train: Matrix of size (N, D)
        - y_train: Vector of size (N,)
        - lr: Learning rate
        - num_iters: Number of iterations of gradient descent to run
        - l2_reg: lambda hyperparameter for using L2 regularization
    Returns:
        - Weight vector w of size (D,)
    """
    N, D = X_train.shape
    ### BEGIN_SOLUTION 1b/1c
    w = np.zeros(D)
    for t in tqdm(range(num_iters)):
        gradient = 0
        ywTx = y_train * (X_train.dot(w))
        sig = sigmoid(-ywTx)
        gradient = 1/N * X_train.T.dot(sig*y_train) - l2_reg * w
        w += lr * (gradient)
    ### END_SOLUTION 1b/1c
    return w

def evaluate(w, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    y_preds = predict(w, X)
    acc = np.mean(y_preds == y)
    print('    {} Accuracy: {}'.format(name, acc))
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=2)
    parser.add_argument('--num-iters', '-T', type=int, default=10000)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot-weights')
    return parser.parse_args()

def read_data(input_df):

    # Shuffle the DataFrame
    df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define features (X) and target (y)
    df = df.drop(columns=['gameid'])
    X = df.drop(columns=['result'])  # Assuming 'result' is the target column
    y = df['result']
    
    # Split the data into train, test, and dev sets (70/10/20 split)
    X_train, X_test_dev, y_train, y_test_dev = train_test_split(X, y, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test_dev, y_test_dev, test_size=0.66, random_state=42)


    return X_train, y_train, X_dev, y_dev, X_test, y_test

def main():
    # Read the data
    input_csv = "../data/2023/2023_LCK_LogReg_Dataset.csv"
    input_df = pd.read_csv(input_csv, index_col=0)
    X_train, y_train, X_dev, y_dev, X_test, y_test = read_data(input_df)

    # all_data = np.load('q1_data.npy', allow_pickle=True).item()
    # X_train = all_data['X_train']
    # y_train = all_data['y_train']
    # X_dev = all_data['X_dev']
    # y_dev = all_data['y_dev']
    # X_test = all_data['X_test']
    # y_test = all_data['y_test']

    # Train with gradient descent
    w = train(X_train, y_train, lr=OPTS.learning_rate, num_iters=OPTS.num_iters, l2_reg=OPTS.l2)

    # Evaluate model
    train_acc = evaluate(w, X_train, y_train, 'Train')
    dev_acc = evaluate(w, X_dev, y_dev, 'Dev')
    if OPTS.test:
        test_acc = evaluate(w, X_test, y_test, 'Test')

    # # Plot the weights
    # if OPTS.plot_weights:
    #     img = w.reshape(28, 28)
    #     vmax = max(np.max(w), -np.min(w))
    #     plt.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
    #     plt.colorbar()
    #     plt.savefig(f'plot_{OPTS.plot_weights}.png')
    #     plt.show()


if __name__ == '__main__':
    OPTS = parse_args()
    main()

