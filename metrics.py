import numpy as np

def rmse(pred: np.ndarray, y: np.ndarray):
    pred = pred.reshape(-1)
    y = y.reshape(-1)
    return np.sqrt(np.mean((pred - y) ** 2))

def mae(pred: np.ndarray, y: np.ndarray):
    pred = pred.reshape(-1)
    y = y.reshape(-1)
    return np.mean(np.abs(pred - y))

def evaluate(pred: np.ndarray, y: np.ndarray):
    return {"rmse": rmse(pred, y), "mae": mae(pred, y)}