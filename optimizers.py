import math
from abc import ABC, abstractmethod
import numpy as np
from interfaces import Params


class Optimizer(ABC):
    def __init__(self, lr:float):
        """Constructor"""
        self.lr = lr
        self.state = {}
        self.t = 0 # This counts the steps
    
    def step(self, params: Params, grads: Params) -> None:
        """This updates the Params and change it"""
        if params.keys() != grads.keys():
            raise KeyError(f"params and grads have different keys: {sorted(params)} vs {sorted(grads)}")
        self.t += 1
        for name in params:
            if params[name].shape != grads[name].shape:
                raise ValueError(f"{name}: param shape {params[name].shape} != grad shape {grads[name].shape}")
            self._update(name, params[name], grads[name])

    def reset(self) -> None:
        """This emptied the state and step counter of the optimizer"""
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

class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, bias_correction=True): # These default parameters align with the one from Pytorch
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        # bias_correction=False with beta_1=0 is exactly RMSProp (alpha=beta_2)
        self.bias_correction = bias_correction

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        # g**2 can overflow float32 (|g| > ~1.8e19) during divergence; clamp v to the largest finite value
        with np.errstate(over="ignore"):
            g_sq = np.square(g)

            if name not in self.state:
                self.state[name] = {"m": (1 - self.beta_1) * g, "v": (1 - self.beta_2) * g_sq}

            else:
                self.state[name] = {"m": self.beta_1 * self.state[name]["m"] + (1 - self.beta_1) * g, "v": self.beta_2 * self.state[name]["v"] + (1 - self.beta_2) * g_sq}

        v = self.state[name]["v"]
        np.minimum(v, np.finfo(v.dtype).max, out=v)

        if self.bias_correction:
            m_hat = self.state[name]["m"]/(1 - self.beta_1**self.t)
            # sqrt(v_hat) computed as sqrt(v)/sqrt(1-beta_2^t) so the division cannot overflow again
            sqrt_v_hat = np.sqrt(v)/math.sqrt(1 - self.beta_2**self.t)
        else:
            m_hat = self.state[name]["m"]
            sqrt_v_hat = np.sqrt(v)

        p -= self.lr * m_hat / (sqrt_v_hat + self.epsilon)


class AdaGrad(Optimizer):
    def __init__(self, lr=1e-3, epsilon=1e-8:
        super().__init__(lr)
        self.epsilon = epsilon

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = np.zeros(p.shape, dtype=p.dtype)
        
        self.state[name] += g * g
        p -= self.lr * g / (np.sqrt(self.state[name]) + self.epsilon)


class RMSProp(Optimizer):
    def __init__(self, lr=1e-3, alpha=0.99, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.alpha = alpha

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = np.zeros(p.shape, dtype=p.dtype)

        self.state[name] = (self.alpha * self.state[name]) + ((1-self.alpha) * g * g)
        p -= self.lr * g / (np.sqrt(self.state[name]) + self.epsilon)
