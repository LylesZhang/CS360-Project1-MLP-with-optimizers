import csv
from pathlib import Path

import numpy as np

from plots import FINAL_STD_DIR, OPTIMIZER_COLORS, SPEED_TARGET, SWEEP_DIR, load_csv, load_summaries

OUT_PATH = Path(__file__).parent / "results.csv"

# winners.csv says 33x for momentum, but a seed diverged at lr 0.003, which sits inside that range
RATIO_OVERRIDES = {"momentum": 3.3}


def read_run(path):
    """(step, seconds) when val loss first reaches SPEED_TARGET or None, and the median time per step"""
    rows = load_csv(path)
    # val loss is only logged at the end of each epoch, so the hit is accurate to one epoch
    hit = next(((int(r["step"]), float(r["time"])) for r in rows
                if r["val_loss"] != "" and float(r["val_loss"]) <= SPEED_TARGET), None)
    # epoch 1 is warm-up; time already excludes validation, so consecutive differences are step costs
    times = [float(r["time"]) for r in rows if int(r["epoch"]) > 1]
    return hit, np.median(np.diff(times))


def evaluate(name, runs, ratio):
    reads = [read_run(FINAL_STD_DIR / f"{r['run_id']}.csv") for r in runs]
    hits = [hit for hit, _ in reads if hit is not None]
    best = [r["best_val_loss"] for r in runs]
    return {
        "optimizer": name,
        "lr": runs[0]["lr"],
        "mean_best_val_loss": round(np.mean(best), 4),
        "sd_best_val_loss": round(np.std(best, ddof=1), 4),
        "steps_to_target": round(np.mean([h[0] for h in hits])) if hits else "not reached",
        "seconds_to_target": round(np.mean([h[1] for h in hits]), 3) if hits else "not reached",
        "reached": f"{len(hits)}/{len(runs)}",
        "n_diverged": sum(r["diverged"] for r in runs),
        "working_ratio": RATIO_OVERRIDES.get(name, ratio),
        "median_step_ms": np.mean([ms for _, ms in reads]) * 1000,
    }


def main():
    summaries = load_summaries(FINAL_STD_DIR / "summary_rows.csv")
    ratios = {r["optimizer"]: float(r["working_ratio"]) for r in load_csv(SWEEP_DIR / "winners.csv")}

    rows = []
    for name in OPTIMIZER_COLORS:
        runs = [s for s in summaries if s["optimizer"] == name]
        if runs:
            rows.append(evaluate(name, runs, ratios.get(name, np.nan)))

    sgd_ms = next((r["median_step_ms"] for r in rows if r["optimizer"] == "sgd"), np.nan)
    for row in rows:
        row["cost_vs_sgd"] = round(row["median_step_ms"] / sgd_ms, 2)
        row["median_step_ms"] = round(row["median_step_ms"], 4)

    with open(OUT_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"speed target: val MSE <= {SPEED_TARGET}")
    for row in rows:
        print(", ".join(f"{key}={value}" for key, value in row.items()))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
