import numpy as np
import pandas as pd

def create_df(filename):
    df = pd.read_csv(filename)
    return df

def read_data(filename):
    x_list = []
    y_list = []
    with open(filename) as f:
        next(f)
        for line in f:
            parts = line.split(',')
            x_list.append([parts[5], parts[6]])
            y_list.append(int(parts[7].rstrip()))
    print(type(parts[7]))
    return np.array(x_list), np.array(y_list)

def predict(X):
    pred = (X[:, 0] > X[:, 1]).astype(int)
    return pred

def evaluate(X, y):
    y_preds = predict(X)
    num_correct = np.sum(y == y_preds)
    total = len(y)
    accuracy = num_correct/total
    print('{} of {} matches correctly predicted.'.format(num_correct, total))
    print('Accuracy = {}'.format(accuracy))
    return accuracy

if __name__ == '__main__':
    X, y = read_data("../data/2023/2023_LCK_match_data_with_winrate.csv")
    evaluate(X, y)
    
