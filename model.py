import numpy as np
from interfaces import Params 

class MLP:
    def __init__(self, layer_sizes: list[int], dtype=np.float32):
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.n_layers = len(layer_sizes) - 1

    def init(self, rng: np.random.Generator) -> dict: 

        params = {}

        for i in range(1, self.n_layers + 1):
            fan_in, fan_out = self.layer_sizes[i-1], self.layer_sizes[i]
            W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0/fan_in)
            b = np.zeros(fan_out)
            params[f"W{i}"] = W.astype(self.dtype)
            params[f"b{i}"] = b.astype(self.dtype)

        return params

    def _forward(self, params: Params, X: np.ndarray):
        A_cache = []
        Z_cache = []
        A = X.astype(self.dtype)
        A_cache.append(A)

        for i in range(1, self.n_layers + 1):
            W_i = params[f"W{i}"]
            b_i = params[f"b{i}"]
            Z = A @ W_i + b_i
            Z_cache.append(Z)
            if i < self.n_layers:
                A = np.maximum(0, Z)
            else:
                A = Z
            A_cache.append(A)

        return A, A_cache, Z_cache

    def loss_and_grads(self, params: Params, X: np.ndarray, y: np.ndarray) -> tuple[float, dict]: 
        y = y.astype(self.dtype).reshape(-1,1)
        out, A_cache, Z_cache = self._forward(params, X)

        n = X.shape[0]
        loss = np.mean((out - y) ** 2)
        dA = 2 * (out - y) / n

        grads = {}
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                dZ = dA
            else:
                dZ = dA * (Z_cache[i-1] > 0)
            A_pre = A_cache[i-1]
            grads[f"W{i}"] = A_pre.T @ dZ
            grads[f"b{i}"] = dZ.sum(axis=0)
            dA = dZ @ params[f"W{i}"].T

        return float(loss), grads

    def predict(self, params: Params, X: np.ndarray) -> np.ndarray: 
        output, A_cache, Z_cache = self._forward(params, X)
        return output