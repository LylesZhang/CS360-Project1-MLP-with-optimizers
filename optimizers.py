import math
from abc import ABC, abstractmethod
import numpy as np
from interfaces import Params


def _clamp_to_finite(a: np.ndarray) -> None:
    #preventing g^2 term from overflowing
    np.minimum(a, np.finfo(a.dtype).max, out=a)


class Optimizer(ABC):
    def __init__(self, lr:float):
        self.lr = lr
        self.state = {}
        self.t = 0 # This counts the steps
    
    def step(self, params: Params, grads: Params) -> None:
        #This updates the Params and change it
        if params.keys() != grads.keys():
            raise KeyError(f"params and grads have different keys: {sorted(params)} vs {sorted(grads)}")
        self.t += 1
        for name in params:
            if params[name].shape != grads[name].shape:
                raise ValueError(f"{name}: param shape {params[name].shape} != grad shape {grads[name].shape}")
            self._update(name, params[name], grads[name])

    # clean the state and reset optimizer
    def reset(self) -> None:
        self.state = {}
        self.t = 0

    @abstractmethod
    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        raise NotImplementedError

class SGD(Optimizer):
    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        p -= self.lr * g

class SGD_momentum(Optimizer):
    def __init__(self, lr, momentum=0.9):
        super().__init__(lr)
        self.m = momentum

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = g.copy()
        else:
            self.state[name] = self.m * self.state[name] + g
        p -= self.lr * self.state[name]


class AdaGrad(Optimizer):
    def __init__(self, lr=1e-3, epsilon=1e-10):
        super().__init__(lr)
        self.epsilon = epsilon

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = np.zeros(p.shape, dtype=p.dtype)

        with np.errstate(over="ignore"):
            self.state[name] += np.square(g)
        # the sum never decays, clamp to avoid overflow
        _clamp_to_finite(self.state[name])
        p -= self.lr * g / (np.sqrt(self.state[name]) + self.epsilon)


class RMSProp(Optimizer):
    def __init__(self, lr=1e-3, alpha=0.99, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.alpha = alpha

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = np.zeros(p.shape, dtype=p.dtype)

        with np.errstate(over="ignore"):
            self.state[name] = (self.alpha * self.state[name]) + ((1-self.alpha) * np.square(g))
        _clamp_to_finite(self.state[name])
        p -= self.lr * g / (np.sqrt(self.state[name]) + self.epsilon)


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, bias_correction=True): # These default parameters align with the one from Pytorch
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction # allowing us to turn off bias correction for degradation check

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        with np.errstate(over="ignore"):
            g_sq = np.square(g)

            if name not in self.state:
                self.state[name] = {"m": (1 - self.beta_1) * g, "v": (1 - self.beta_2) * g_sq}

            else:
                self.state[name] = {"m": self.beta_1 * self.state[name]["m"] + (1 - self.beta_1) * g, "v": self.beta_2 * self.state[name]["v"] + (1 - self.beta_2) * g_sq}

        v = self.state[name]["v"]
        _clamp_to_finite(v)

        if self.bias_correction:
            m_hat = self.state[name]["m"]/(1 - self.beta_1**self.t)
            sqrt_v_hat = np.sqrt(v)/math.sqrt(1 - self.beta_2**self.t)
        else:
            m_hat = self.state[name]["m"]
            sqrt_v_hat = np.sqrt(v)

        p -= self.lr * m_hat / (sqrt_v_hat + self.epsilon)

