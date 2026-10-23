import numpy as np
import torch
from torch import nn
from model import MLP

def test_torch_parity():
    m = MLP(layer_sizes=[4, 5, 5, 1], dtype=np.float64)
    rng = np.random.default_rng(0)
    params = m.init(rng)
    X = np.asarray(rng.standard_normal((6,4)))
    y = np.asarray(rng.standard_normal((6,1)))
    loss, grads = m.loss_and_grads(params,X, y)

    torch_model = nn.Sequential(
        nn.Linear(4,5),
        nn.ReLU(),
        nn.Linear(5,5),
        nn.ReLU(),
        nn.Linear(5,1)
    ).double()

    with torch.no_grad():
        for i in [0, 2, 4]:
            torch_model[i].weight.copy_(torch.from_numpy(params[f"W{i//2 +1}"].T))
            torch_model[i].bias.copy_(torch.from_numpy(params[f"b{i//2 +1}"]))

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).reshape(-1,1)

    out_t = torch_model(X_t)
    loss_t = nn.MSELoss()(out_t, y_t)
    loss_t.backward()

    np.testing.assert_allclose(m.predict(params, X), out_t.detach().numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(loss, loss_t.item(), rtol=1e10, atol=1e-12)

    for i in [0, 2, 4]:
        np.testing.assert_allclose(grads[f"W{i//2+1}"], torch_model[i].weight.grad.numpy().T, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grads[f"b{i//2+1}"], torch_model[i].bias.grad.numpy(), rtol=1e-10, atol=1e-12)

if __name__ == "__main__":
    test_torch_parity()
    print("Torch parity check passed")