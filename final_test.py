import csv

import numpy as np

from data import prepare_data
from metrics import rmse, mae
from model import MLP
from train import LOG_DIR, LAYER_SIZES

FINAL_DIR = LOG_DIR / "final_std"

def load_csv(path):
    with open(path, newline="") as csvfile:
        return list(csv.DictReader(csvfile))

def write_csv(path, rows):
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def load_params(path):
    loaded = np.load(path)
    params = {}
    for name in loaded.files:
        params[name] = loaded[name]
    return params

def main():
    d = prepare_data(seed=0, standardize=True)
    model = MLP(LAYER_SIZES, dtype=np.float32)

    per_run = []
    for r in load_csv(FINAL_DIR / "summary_rows.csv"):
        params = load_params(FINAL_DIR / f"{r["run_id"]}.npz")
        pred = model.predict(params, d["X_test"])
        per_run.append({
            "run_id":r["run_id"],
            "optimizer": r["optimizer"],
            "lr": r["lr"],
            "seed": r["seed"],
            "test_rmse": rmse(pred, d["y_test"]),
            "test_mae": mae(pred, d["y_test"]),

        })
    write_csv(LOG_DIR/"test_results.csv",per_run)


    names = []
    for r in per_run:
        if r["optimizer"] not in names:
            names.append(r["optimizer"])

    summary = []
    for name in names:
        rmses = []
        maes = []
        for r in per_run:
            if r["optimizer"] == name:
                rmses.append(r["test_rmse"])
                maes.append(r["test_mae"])
        rmses = np.array(rmses)
        maes = np.array(maes)

        summary.append({
                "model": name,
                "n_runs": len(rmses),
                "test_rmse_mean": rmses.mean(),
                "test_rmse_sd": rmses.std(ddof=1),
                "test_mae_mean": maes.mean(),
                "test_mae_sd": maes.std(ddof=1),
            })

    for baseline in load_csv(LOG_DIR / "baselines.csv"):
        summary.append({
            "model": baseline["model"],
            "n_runs": 1,
            "test_rmse_mean": float(baseline["test_rmse"]),
            "test_rmse_sd": "",
            "test_mae_mean": float(baseline["test_mae"]),
            "test_mae_sd": "",
        })
    write_csv(LOG_DIR / "test_summary.csv", summary)

    print(f"Wrote {LOG_DIR/"test_results.csv"} and {LOG_DIR/"test_summary.csv"}")

if __name__ == "__main__":
    main()