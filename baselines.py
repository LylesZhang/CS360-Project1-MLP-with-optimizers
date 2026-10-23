import argparse

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from data import prepare_data


def rmse(y, y_pred):
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


def mae(y, y_pred):
    return float(np.mean(np.abs(y - y_pred)))


def evaluate(predict, d):
    """RMSE / MAE on train, val and test for a function X -> y_pred"""
    out = {}
    for split in ("train", "val", "test"):
        y, y_pred = d[f"y_{split}"], np.ravel(predict(d[f"X_{split}"]))
        out[f"{split}_rmse"] = rmse(y, y_pred)
        out[f"{split}_mae"] = mae(y, y_pred)
    return out


def mean_baseline(d):
    """Always predict the training-set mean: the score any useful model must beat"""
    mu = d["y_train"].mean()
    return evaluate(lambda X: np.full(len(X), mu), d)


def closed_form_linear(d):
    """Least squares w = argmin ||[X 1] w - y||^2, used to cross-check sklearn"""
    def add_bias(X):
        return np.hstack([X, np.ones((len(X), 1))])
    w, *_ = np.linalg.lstsq(add_bias(d["X_train"]), d["y_train"], rcond=None)
    return evaluate(lambda X: add_bias(X) @ w, d)


def sklearn_linear(d):
    lr = LinearRegression().fit(d["X_train"], d["y_train"])
    return evaluate(lr.predict, d)


def sklearn_mlp(d, hidden, seed):
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                       learning_rate_init=1e-3, batch_size=64, max_iter=200,
                       early_stopping=True, random_state=seed)
    mlp.fit(d["X_train"], d["y_train"])
    return evaluate(mlp.predict, d)


def main():
    parser = argparse.ArgumentParser(description="Baselines that set the RMSE ballpark for our MLP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    args = parser.parse_args()

    d = prepare_data(seed=0, standardize=True)
    results = {
        "mean": mean_baseline(d),
        "linear (closed form)": closed_form_linear(d),
        "linear (sklearn)": sklearn_linear(d),
        f"MLPRegressor {tuple(args.hidden)}": sklearn_mlp(d, tuple(args.hidden), args.seed),
    }

    print(f"{'model':<28}{'val RMSE':>10}{'val MAE':>10}{'test RMSE':>11}{'test MAE':>10}")
    for name, r in results.items():
        print(f"{name:<28}{r['val_rmse']:>10.4f}{r['val_mae']:>10.4f}{r['test_rmse']:>11.4f}{r['test_mae']:>10.4f}")


if __name__ == "__main__":
    main()
