from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import numpy as np

data_dir = Path(__file__).parent/"data"

def load_data():
    housing = fetch_california_housing(data_home=data_dir)
    return housing.data, housing.target

def split_indices(n, seed=0):
    ids = np.arange(n)
    train_ids, remainder_ids = train_test_split(ids, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(remainder_ids, test_size=0.50, random_state=seed)

    return train_ids, val_ids, test_ids

def fit_standardizer(X):

    X_values = np.asarray(X, dtype=float)
    constant = np.all(X_values == X_values[0], axis=0)
    means = X_values.mean(axis=0)
    scales = X_values.std(axis=0, ddof=0)
    means[constant] = X_values[0, constant]
    scales[constant] = 1.0
    return means, scales


def apply_standardizer(X, means, scales):

    X_values = np.asarray(X, dtype=float)
    return (X_values - means) / scales


def prepare_data(seed=0, standardize=True):

    X, y = load_data()
    train_ids, val_ids, test_ids = split_indices(len(X), seed=seed)

    X_train, X_val, X_test = X[train_ids], X[val_ids], X[test_ids]

    if standardize:
        means, scales = fit_standardizer(X_train)
        X_train = apply_standardizer(X_train, means, scales)
        X_val = apply_standardizer(X_val, means, scales)
        X_test = apply_standardizer(X_test, means, scales)

    return {
        "X_train": X_train, "y_train": y[train_ids],
        "X_val": X_val, "y_val": y[val_ids],
        "X_test": X_test, "y_test": y[test_ids],
    }

if __name__ == "__main__":
    d = prepare_data()
    for k, v in d.items():
        print(k, v.shape)