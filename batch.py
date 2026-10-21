import numpy as np 

def make_batches(X, y, batch_size, rng):
    n = len(X)
    ids = np.arange(n)
    rng.shuffle(ids)
    batches= []

    for i in range(0, n, batch_size):
        batch_ids = ids[i:i+batch_size]
        batches.append((X[batch_ids], y[batch_ids]))

    return batches