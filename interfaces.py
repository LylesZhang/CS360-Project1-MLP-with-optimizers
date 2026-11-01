from typing import Protocol

import numpy as np

# name -> array; grads use the same keys and shapes. Optimizers update params in place.
Params = dict[str, np.ndarray]

class OptimizerAPI(Protocol):
    def step(self, params: Params, grads: Params) -> None:
        """Update every params[name] in place using grads[name]"""
    def reset(self) -> None:
        """Clear all internal state (moments) and the step counter"""



