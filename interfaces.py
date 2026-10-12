from typing import Protocol

import numpy as np

Params = dict[str, np.ndarray]

class Optimizer:
    def __init__(self, lr:float):
        """Constructor"""    
    def step(self, params: Params, grads: Params) -> None:
        """This updates the Params and change it"""
    def reset(self) -> None:
        """This emptied the Params"""
