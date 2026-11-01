import numpy as np
import pytest
import torch
from torch import nn

from batch import make_batches
from model import MLP
from optimizers import SGD, SGD_momentum, Adam, AdaGrad, RMSProp
from torch_reference import build_torch_mlp

LAYER_SIZES = [4, 8, 8, 1]
N_SAMPLES = 32
BATCH_SIZE = 8
N_EPOCHS = 3
TOL = 1e-8

# (ours, torch) pairs; hyperparameters spelled out since some defaults differ
OPTIMIZER_PAIRS = {
    "SGD": (lambda: SGD(0.1),
            lambda ps: torch.optim.SGD(ps, lr=0.1)),
    "Momentum": (lambda: SGD_momentum(0.1, momentum=0.9),
                 lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9)),
    "Adam": (lambda: Adam(1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
             lambda ps: torch.optim.Adam(ps, lr=1e-2, betas=(0.9, 0.999), eps=1e-8)),
    "AdaGrad": (lambda: AdaGrad(1e-1, epsilon=1e-10),
                lambda ps: torch.optim.Adagrad(ps, lr=1e-1, eps=1e-10)),
    "RMSProp": (lambda: RMSProp(1e-2, alpha=0.99, epsilon=1e-8),
                lambda ps: torch.optim.RMSprop(ps, lr=1e-2, alpha=0.99, eps=1e-8)),
}


@pytest.mark.parametrize("name", OPTIMIZER_PAIRS)
def test_training_matches_torch(name):
    make_ours, make_torch = OPTIMIZER_PAIRS[name]

    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, LAYER_SIZES[0]))
    y = rng.standard_normal(N_SAMPLES)

    m = MLP(LAYER_SIZES, dtype=np.float64)
    params = m.init(rng)
    torch_model = build_torch_mlp(LAYER_SIZES, params)

    opt = make_ours()
    opt_t = make_torch(torch_model.parameters())

    # both sides train on the exact same batches
    batches = []
    for _ in range(N_EPOCHS):
        batches += make_batches(X, y, BATCH_SIZE, rng)

    losses, losses_t = [], []
    for Xb, yb in batches:
        loss, grads = m.loss_and_grads(params, Xb, yb)
        opt.step(params, grads)
        losses.append(loss)

        # torch accumulates grads, so clear them first
        opt_t.zero_grad()
        loss_t = nn.MSELoss()(torch_model(torch.from_numpy(Xb)), torch.from_numpy(yb).reshape(-1, 1))
        loss_t.backward()
        opt_t.step()
        losses_t.append(loss_t.item())

    np.testing.assert_allclose(losses, losses_t, rtol=TOL, atol=TOL)

    # losses miss the last update, so check the final params too
    for k in range(1, len(LAYER_SIZES)):
        layer = torch_model[2 * (k-1)]
        np.testing.assert_allclose(params[f"W{k}"], layer.weight.detach().numpy().T, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(params[f"b{k}"], layer.bias.detach().numpy(), rtol=TOL, atol=TOL)
