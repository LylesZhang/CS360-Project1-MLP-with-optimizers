import numpy as np
from model import MLP

def test_grad_check(eps=1e-6):
    m = MLP(layer_sizes=[4, 5, 5, 1], dtype=np.float64)
    rng = np.random.default_rng(0)
    params = m.init(rng)
    X = np.asarray(rng.standard_normal((6,4)))
    y = np.asarray(rng.standard_normal((6,1)))

    loss, grads = m.loss_and_grads(params,X, y)

    for i in params:
        num_grad = np.zeros_like(params[i])
        for idx in np.ndindex(params[i].shape):
            old_val = params[i][idx]

            params[i][idx] = old_val + eps
            loss_plus, _ = m.loss_and_grads(params,X, y)

            params[i][idx] = old_val - eps
            loss_minus, _ = m.loss_and_grads(params,X, y)

            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

            params[i][idx] = old_val

        diff = np.linalg.norm(num_grad - grads[i])
        scale = np.linalg.norm(num_grad) + np.linalg.norm(grads[i])
        assert diff / scale <1e-6

if __name__ == "__main__":
    test_grad_check()
    print("All gradient checks passed")