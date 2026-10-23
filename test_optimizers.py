import numpy as np
import pytest
import torch

from optimizers import SGD, SGD_momentum, Adam

SHAPES = {"W1": (4, 5), "b1": (5,), "W2": (5, 1), "b2": (1,)}
N_STEPS = 50
TOL = 1e-6


def make_problem(seed=0):
    """Random float64 initial params and a fixed sequence of random gradients"""
    rng = np.random.default_rng(seed)
    params = {k: rng.standard_normal(s) for k, s in SHAPES.items()}
    grad_seq = [{k: rng.standard_normal(s) for k, s in SHAPES.items()} for _ in range(N_STEPS)]
    return params, grad_seq


def run_numpy(opt, params, grad_seq):
    params = {k: v.copy() for k, v in params.items()}
    history = []
    for grads in grad_seq:
        opt.step(params, grads)
        history.append({k: v.copy() for k, v in params.items()})
    return history


def run_torch(make_opt, params, grad_seq):
    tensors = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in params.items()}
    opt = make_opt(list(tensors.values()))
    history = []
    for grads in grad_seq:
        for k, t in tensors.items():
            t.grad = torch.tensor(grads[k], dtype=torch.float64)
        opt.step()
        history.append({k: t.detach().numpy().copy() for k, t in tensors.items()})
    return history


def assert_same_trajectory(hist_a, hist_b, tol=TOL):
    for step, (a, b) in enumerate(zip(hist_a, hist_b), start=1):
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=tol, atol=tol, err_msg=f"step {step}, param {k}")


# ---------- torch parity ----------

@pytest.mark.parametrize("lr", [0.1, 0.01])
def test_sgd_parity(lr):
    params, grad_seq = make_problem()
    ours = run_numpy(SGD(lr), params, grad_seq)
    ref = run_torch(lambda ps: torch.optim.SGD(ps, lr=lr), params, grad_seq)
    assert_same_trajectory(ours, ref)


@pytest.mark.parametrize("lr,momentum", [(0.1, 0.9), (0.01, 0.5)])
def test_momentum_parity(lr, momentum):
    params, grad_seq = make_problem()
    ours = run_numpy(SGD_momentum(lr, momentum), params, grad_seq)
    ref = run_torch(lambda ps: torch.optim.SGD(ps, lr=lr, momentum=momentum), params, grad_seq)
    assert_same_trajectory(ours, ref)


@pytest.mark.parametrize("lr,betas,eps", [(1e-3, (0.9, 0.999), 1e-8), (1e-2, (0.8, 0.99), 1e-6)])
def test_adam_parity(lr, betas, eps):
    params, grad_seq = make_problem()
    ours = run_numpy(Adam(lr, betas[0], betas[1], eps), params, grad_seq)
    ref = run_torch(lambda ps: torch.optim.Adam(ps, lr=lr, betas=betas, eps=eps), params, grad_seq)
    assert_same_trajectory(ours, ref)


# ---------- equivalence ----------

def test_momentum_zero_equals_sgd():
    params, grad_seq = make_problem()
    with_mu0 = run_numpy(SGD_momentum(0.1, momentum=0.0), params, grad_seq)
    plain = run_numpy(SGD(0.1), params, grad_seq)
    assert_same_trajectory(with_mu0, plain, tol=0)


@pytest.mark.parametrize("alpha,eps", [(0.99, 1e-8), (0.9, 1e-6)])
def test_adam_degrades_to_rmsprop(alpha, eps):
    params, grad_seq = make_problem()
    ours = run_numpy(Adam(1e-2, beta_1=0.0, beta_2=alpha, epsilon=eps, bias_correction=False), params, grad_seq)
    ref = run_torch(lambda ps: torch.optim.RMSprop(ps, lr=1e-2, alpha=alpha, eps=eps), params, grad_seq)
    assert_same_trajectory(ours, ref)


# ---------- Adam first step ----------

def test_adam_first_step_moves_by_lr():
    # step 1: m_hat = g, v_hat = g^2, so the update is lr * g / (|g| + eps) ~= lr * sign(g)
    params, grad_seq = make_problem()
    lr = 1e-3
    before = {k: v.copy() for k, v in params.items()}
    Adam(lr).step(params, grad_seq[0])
    eps = 1e-8
    for k in params:
        g = grad_seq[0][k]
        delta = params[k] - before[k]
        np.testing.assert_allclose(np.abs(delta), lr * np.abs(g) / (np.abs(g) + eps), rtol=1e-10)
        np.testing.assert_allclose(np.abs(delta), lr, rtol=1e-4)
        assert np.all(np.sign(delta) == -np.sign(grad_seq[0][k]))


# ---------- base class behaviour ----------

def test_reset_clears_state():
    params, grad_seq = make_problem()
    opt = Adam(1e-3)
    run_numpy(opt, params, grad_seq[:5])
    opt.reset()
    assert opt.t == 0 and opt.state == {}
    # after reset the optimizer must behave exactly like a brand-new one
    assert_same_trajectory(run_numpy(opt, params, grad_seq), run_numpy(Adam(1e-3), params, grad_seq), tol=0)


def test_updates_in_place():
    params, grad_seq = make_problem()
    W1 = params["W1"]
    SGD(0.1).step(params, grad_seq[0])
    assert params["W1"] is W1


def test_mismatched_grads_raise():
    params, grad_seq = make_problem()
    with pytest.raises(KeyError):
        SGD(0.1).step(params, {"W1": grad_seq[0]["W1"]})
    bad = dict(grad_seq[0], W1=np.zeros((5, 4)))
    with pytest.raises(ValueError):
        SGD(0.1).step(params, bad)


def test_adam_float32_overflow_guard():
    p = {"w": np.zeros(3, dtype=np.float32)}
    g = {"w": np.array([1e20, -1e20, 1.0], dtype=np.float32)}  # 1e40 overflows float32
    Adam(1e-3).step(p, g)
    assert np.all(np.isfinite(p["w"]))
