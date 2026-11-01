import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from metrics import rmse, mae, evaluate


def make_pred_and_y(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal(n)


def test_hand_computed():
    pred = np.array([0.0, 0.0])
    y = np.array([4.0, 0.0])
    assert rmse(pred, y) == pytest.approx(np.sqrt(8))
    assert mae(pred, y) == pytest.approx(2.0)


def test_matches_sklearn():
    pred, y = make_pred_and_y()
    assert rmse(pred, y) == pytest.approx(root_mean_squared_error(y, pred))
    assert mae(pred, y) == pytest.approx(mean_absolute_error(y, pred))


def test_column_pred_and_flat_y():
    # predict() returns (n, 1) while the data's y is (n,); without the reshape this broadcasts to (n, n)
    pred, y = make_pred_and_y()
    assert rmse(pred.reshape(-1, 1), y) == pytest.approx(rmse(pred, y))
    assert mae(pred.reshape(-1, 1), y) == pytest.approx(mae(pred, y))


def test_evaluate_returns_both_metrics():
    pred, y = make_pred_and_y()
    result = evaluate(pred, y)
    assert set(result) == {"rmse", "mae"}
    assert result["rmse"] == pytest.approx(rmse(pred, y))
    assert result["mae"] == pytest.approx(mae(pred, y))
