import time

import numpy as np

from batch import make_batches
from model import MLP

def mse(model, params, X, y):
    predictions = model.predict(params, X).reshape(-1)
    return float(np.mean((predictions - y) ** 2))

