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
        self.t += 1
        for name in params:
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
    def __init__(self, lr, momentum):
        super().__init__(lr)
        self.m = momentum

    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = g.copy()
        else:
            self.state[name] = self.m * self.state[name] + g
        p -= self.lr * self.state[name]

class Adam(Optimizer):
    def __init__(self, lr, beta_1, beta_2, epsilon):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
    
    def _update(self, name:str, p: np.ndarray, g: np.ndarray):
        if name not in self.state:
            self.state[name] = {"m": (1 - self.beta_1) * g.copy(), "v": (1 - self.beta_2) * np.square(g.copy())}

        else:
            self.state[name] = {"m": self.beta_1 * self.state[name]["m"] + (1 - self.beta_1) * g.copy(), "v": self.beta_2 * self.state[name]["v"] + (1 - self.beta_2) * np.square(g.copy())}

        m_hat = self.state[name]["m"]/(1 - self.beta_1**self.t)
        v_hat = self.state[name]["v"]/(1 - self.beta_2**self.t)

        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        



