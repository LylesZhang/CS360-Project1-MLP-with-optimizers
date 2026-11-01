import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from data import prepare_data
from metrics import rmse, mae

LOG_DIR = Path(__file__).parent/"logs"
COLUMNS = ["model", "train_rmse", "train_mae", "val_rmse", "val_mae", "test_rmse", "test_mae"]


def evaluate_splits(predict, d):
    """RMSE / MAE on train, val and test for a function X -> y_pred"""
    out = {}
    for split in ("train", "val", "test"):
        y, y_pred = d[f"y_{split}"], np.ravel(predict(d[f"X_{split}"]))
        out[f"{split}_rmse"] = rmse(y_pred, y)
        out[f"{split}_mae"] = mae(y_pred, y)
    return out


def mean_baseline(d):
    """Always predict the training-set mean: the score any useful model must beat"""
    mu = d["y_train"].mean()
    return evaluate_splits(lambda X: np.full(len(X), mu), d)


def closed_form_linear(d):
    """Least squares w = argmin ||[X 1] w - y||^2, used to cross-check sklearn"""
    def add_bias(X):
        return np.hstack([X, np.ones((len(X), 1))])
    w, *_ = np.linalg.lstsq(add_bias(d["X_train"]), d["y_train"], rcond=None)
    return evaluate_splits(lambda X: add_bias(X) @ w, d)


def sklearn_linear(d):
    lr = LinearRegression().fit(d["X_train"], d["y_train"])
    return evaluate_splits(lr.predict, d)


def sklearn_mlp(d, hidden, seed):
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                       learning_rate_init=1e-3, batch_size=64, max_iter=200,
                       early_stopping=True, random_state=seed)
    mlp.fit(d["X_train"], d["y_train"])
    return evaluate_splits(mlp.predict, d)


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


    rows = []
    for name, r in results.items():
        row = {"model": name}
        for key in COLUMNS[1:]:
            row[key] = r[key]
        rows.append(row)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "baselines.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {LOG_DIR / "baselines.csv"}")


if __name__ == "__main__":
    main()
