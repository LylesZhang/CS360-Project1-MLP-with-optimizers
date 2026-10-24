import numpy as np
import pytest
import torch

from optimizers import SGD, SGD_momentum, Adam, AdaGrad, RMSProp

SHAPES = {"W1": (4, 5), "b1": (5,), "W2": (5, 1), "b2": (1,)}
N_STEPS = 50
TOL = 1e-6

# fresh instance of every optimizer, for tests of shared base-class behaviour
ALL_OPTIMIZERS = {
    "SGD": lambda: SGD(0.1),
    "Momentum": lambda: SGD_momentum(0.1, 0.9),
    "Adam": lambda: Adam(1e-2),
    "AdaGrad": lambda: AdaGrad(1e-2, 1e-10),
    "RMSProp": lambda: RMSProp(1e-2, 0.99, 1e-8),
}
# optimizers that keep a running sum/average of g^2
SECOND_MOMENT_OPTIMIZERS = ["Adam", "AdaGrad", "RMSProp"]


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


def state_arrays(opt):
    """Every ndarray inside opt.state (SGD_momentum/AdaGrad/RMSProp store arrays, Adam stores dicts of arrays)"""
    for s in opt.state.values():
        yield from (s.values() if isinstance(s, dict) else [s])


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


# our AdaGrad defaults (lr=1e-3, eps=1e-8) differ from torch's (lr=1e-2, eps=1e-10), so always pass them explicitly
@pytest.mark.parametrize("lr,eps", [(1e-2, 1e-10), (1e-1, 1e-8)])
def test_adagrad_parity(lr, eps):
    params, grad_seq = make_problem()
    ours = run_numpy(AdaGrad(lr, eps), params, grad_seq)
    ref = run_torch(lambda ps: torch.optim.Adagrad(ps, lr=lr, eps=eps), params, grad_seq)
    assert_same_trajectory(ours, ref)


@pytest.mark.parametrize("lr,alpha,eps", [(1e-2, 0.99, 1e-8), (1e-3, 0.9, 1e-6)])
def test_rmsprop_parity(lr, alpha, eps):
    params, grad_seq = make_problem()
    ours = run_numpy(RMSProp(lr, alpha, eps), params, grad_seq)
    ref = run_torch(lambda ps: torch.optim.RMSprop(ps, lr=lr, alpha=alpha, eps=eps), params, grad_seq)
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


@pytest.mark.parametrize("alpha,eps", [(0.99, 1e-8), (0.9, 1e-6)])
def test_rmsprop_class_equals_adam_degenerate(alpha, eps):
    # two code paths for the same algorithm; not tol=0 because the floating-point operation order differs
    params, grad_seq = make_problem()
    ours = run_numpy(RMSProp(1e-2, alpha, eps), params, grad_seq)
    adam = run_numpy(Adam(1e-2, beta_1=0.0, beta_2=alpha, epsilon=eps, bias_correction=False), params, grad_seq)
    assert_same_trajectory(ours, adam, tol=1e-12)


# ---------- defining properties of AdaGrad / RMSProp ----------

def test_adagrad_first_step_moves_by_lr():
    # step 1: sum = g^2, so the update is lr * g / (|g| + eps) ~= lr * sign(g)
    params, grad_seq = make_problem()
    lr, eps = 1e-3, 1e-10
    before = {k: v.copy() for k, v in params.items()}
    AdaGrad(lr, eps).step(params, grad_seq[0])
    for k in params:
        g = grad_seq[0][k]
        delta = params[k] - before[k]
        np.testing.assert_allclose(np.abs(delta), lr * np.abs(g) / (np.abs(g) + eps), rtol=1e-10)
        np.testing.assert_allclose(np.abs(delta), lr, rtol=1e-6)
        assert np.all(np.sign(delta) == -np.sign(g))


@pytest.mark.parametrize("alpha", [0.99, 0.9])
def test_rmsprop_first_step_is_lr_over_sqrt_one_minus_alpha(alpha):
    # no bias correction: v_1 = (1-alpha) g^2, so the update is lr * g / (sqrt(1-alpha) |g| + eps) ~= lr / sqrt(1-alpha)
    params, grad_seq = make_problem()
    lr, eps = 1e-3, 1e-8
    before = {k: v.copy() for k, v in params.items()}
    RMSProp(lr, alpha, eps).step(params, grad_seq[0])
    for k in params:
        g = grad_seq[0][k]
        delta = params[k] - before[k]
        np.testing.assert_allclose(np.abs(delta), lr * np.abs(g) / (np.sqrt(1 - alpha) * np.abs(g) + eps), rtol=1e-10)
        np.testing.assert_allclose(np.abs(delta), lr / np.sqrt(1 - alpha), rtol=1e-4)
        assert np.all(np.sign(delta) == -np.sign(g))


def test_adagrad_step_decays_as_inv_sqrt_t():
    # constant g: sum_t = t g^2, so |delta_t| = lr |g| / (sqrt(t) |g| + eps) ~= lr / sqrt(t)
    lr, eps, g = 0.1, 1e-10, 0.5
    opt = AdaGrad(lr, eps)
    p = {"w": np.zeros(1)}
    for t in range(1, 201):
        before = p["w"].copy()
        opt.step(p, {"w": np.array([g])})
        np.testing.assert_allclose(np.abs(p["w"] - before), lr * g / (np.sqrt(t) * g + eps), rtol=1e-10)
    np.testing.assert_allclose(np.abs(p["w"] - before), lr / np.sqrt(200), rtol=1e-6)


def test_rmsprop_constant_grad_steady_state():
    # constant g: v_t = (1-alpha^t) g^2, so |delta_t| = lr / sqrt(1-alpha^t) -> lr (does NOT decay, unlike AdaGrad)
    lr, alpha, eps, g = 0.1, 0.99, 1e-12, 0.5
    opt = RMSProp(lr, alpha, eps)
    p = {"w": np.zeros(1)}
    for t in range(1, 3001):
        before = p["w"].copy()
        opt.step(p, {"w": np.array([g])})
        np.testing.assert_allclose(np.abs(p["w"] - before), lr * g / (np.sqrt(1 - alpha**t) * g + eps), rtol=1e-8)
    np.testing.assert_allclose(np.abs(p["w"] - before), lr, rtol=1e-8)


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

@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_reset_clears_state(name):
    params, grad_seq = make_problem()
    opt = ALL_OPTIMIZERS[name]()
    run_numpy(opt, params, grad_seq[:5])
    opt.reset()
    assert opt.t == 0 and opt.state == {}
    # after reset the optimizer must behave exactly like a brand-new one
    assert_same_trajectory(run_numpy(opt, params, grad_seq), run_numpy(ALL_OPTIMIZERS[name](), params, grad_seq), tol=0)


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_updates_in_place(name):
    params, grad_seq = make_problem()
    originals = dict(params)
    opt = ALL_OPTIMIZERS[name]()
    for grads in grad_seq[:3]:
        opt.step(params, grads)
    for k in params:
        assert params[k] is originals[k]


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_keeps_float32(name):
    # experiments run in float32: neither params nor optimizer state may be silently upcast to float64
    rng = np.random.default_rng(0)
    params = {k: rng.standard_normal(s).astype(np.float32) for k, s in SHAPES.items()}
    opt = ALL_OPTIMIZERS[name]()
    for _ in range(3):
        opt.step(params, {k: rng.standard_normal(s).astype(np.float32) for k, s in SHAPES.items()})
    assert all(p.dtype == np.float32 for p in params.values())
    assert all(s.dtype == np.float32 for s in state_arrays(opt))


@pytest.mark.parametrize("name", ALL_OPTIMIZERS)
def test_state_does_not_alias_grads(name):
    # the caller may reuse its grad buffers; the optimizer's state must not change with them
    params, grad_seq = make_problem()
    opt = ALL_OPTIMIZERS[name]()
    grads = {k: v.copy() for k, v in grad_seq[0].items()}
    opt.step(params, grads)
    snapshot = [s.copy() for s in state_arrays(opt)]
    for g in grads.values():
        g[...] = 1e6
    for s, s0 in zip(state_arrays(opt), snapshot):
        np.testing.assert_array_equal(s, s0)


def test_mismatched_grads_raise():
    params, grad_seq = make_problem()
    with pytest.raises(KeyError):
        SGD(0.1).step(params, {"W1": grad_seq[0]["W1"]})
    bad = dict(grad_seq[0], W1=np.zeros((5, 4)))
    with pytest.raises(ValueError):
        SGD(0.1).step(params, bad)


# ---------- float32 overflow guards ----------

@pytest.mark.parametrize("name", SECOND_MOMENT_OPTIMIZERS)
def test_float32_overflow_guard(name):
    # 1e21 is a finite float32, but (1e21)^2 = 1e42 overflows it (max ~3.4e38), as does (1-alpha) * 1e42
    opt = ALL_OPTIMIZERS[name]()
    p = {"w": np.zeros(3, dtype=np.float32)}
    with np.errstate(over="raise", invalid="raise"):  # any un-silenced overflow or inf-inf/inf/inf fails the test
        opt.step(p, {"w": np.array([1e21, -1e21, 1.0], dtype=np.float32)})
        for _ in range(10):  # back to ordinary gradients
            opt.step(p, {"w": np.ones(3, dtype=np.float32)})
    assert np.all(np.isfinite(p["w"]))
    assert all(np.all(np.isfinite(s)) for s in state_arrays(opt))


@pytest.mark.parametrize("name", ["Adam", "RMSProp"])
def test_clamped_moving_average_decays_back(name):
    # with the clamp, an exponential moving average of g^2 shrinks again once gradients are normal
    # (without it, beta * inf = inf would freeze the parameter forever)
    def g_sq_average(opt):
        s = opt.state["w"]
        return s["v"] if isinstance(s, dict) else s

    opt = ALL_OPTIMIZERS[name]()
    p = {"w": np.zeros(1, dtype=np.float32)}
    opt.step(p, {"w": np.array([1e21], dtype=np.float32)})
    peak = g_sq_average(opt).copy()
    assert np.all(np.isfinite(peak))
    for _ in range(100):
        opt.step(p, {"w": np.ones(1, dtype=np.float32)})
    assert np.all(g_sq_average(opt) < peak)
