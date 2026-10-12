import numpy as np


class Optimizer:
    """优化器基类。params 和 grads 都是 dict: name -> np.ndarray，key 和 shape 一一对应。
    step() 原地更新 params。"""

    def __init__(self, lr):
        self.lr = lr
        self.state = {}  # name -> 该参数自己的状态（动量缓存、m/v 等）
        self.t = 0       # 已执行的步数

    def step(self, params, grads):
        self.t += 1
        for name in params:
            self._update(name, params[name], grads[name])

    def _update(self, name, p, g):
        """子类实现：用梯度 g 原地更新参数 p。"""
        raise NotImplementedError

    def reset(self):
        self.state = {}
        self.t = 0


class SGD(Optimizer):
    def _update(self, name, p, g):
        p -= self.lr * g  # 原地修改，调用方持有的数组也会变


class Momentum(Optimizer):
    """PyTorch 写法：v = mu*v + g; p -= lr*v"""

    def __init__(self, lr, mu=0.9):
        super().__init__(lr)
        self.mu = mu

    def _update(self, name, p, g):
        if name not in self.state:
            self.state[name] = g.copy()  # torch：第一步 buf 直接等于 g
        else:
            v = self.state[name]
            v *= self.mu
            v += g
        p -= self.lr * self.state[name]


class Adam(Optimizer):
    """PyTorch 写法：eps 在根号外面。bias_correction=False 且 beta1=0 时等价于 RMSprop。"""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
        super().__init__(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.bias_correction = bias_correction

    def _update(self, name, p, g):
        if name not in self.state:
            self.state[name] = {"m": np.zeros_like(g), "v": np.zeros_like(g)}
        s = self.state[name]
        s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * g
        s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * g * g
        if self.bias_correction:
            m_hat = s["m"] / (1 - self.beta1 ** self.t)
            v_hat = s["v"] / (1 - self.beta2 ** self.t)
        else:
            m_hat, v_hat = s["m"], s["v"]
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
