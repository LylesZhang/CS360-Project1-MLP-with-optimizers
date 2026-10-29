import csv

import numpy as np

from data import prepare_data
from train import LOG_DIR, OPTIMIZERS, training_run

SWEEP_DIR = LOG_DIR / "lr_sweep"
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SEEDS = [0, 1, 2, 3, 4]
N_EPOCHS = 20


def constant_baseline_loss(d):
    # the best any constant prediction can do on val is var(y_val), so a network that died and outputs a constant never gets below it
    return np.var(d["y_val"])


def run_sweep(d):
    # training_run appends to summary_rows.csv, so clear it first or a rerun would duplicate every row
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "summary_rows.csv").unlink(missing_ok=True)

    summaries = []
    for name in OPTIMIZERS:
        print(f"running {name}")
        for lr in LRS:
            for seed in SEEDS:
                # large learning rates are expected to overflow, training_run already records that as diverged
                with np.errstate(all="ignore"):
                    summary, _ = training_run(d, name, lr, seed, N_EPOCHS, log_dir=SWEEP_DIR)
                summaries.append(summary)
    return summaries


def summarize(summaries, baseline):
    rows = []
    for name in OPTIMIZERS:
        for lr in LRS:
            runs = [s for s in summaries if s["optimizer"] == name and s["lr"] == lr]
            losses = np.array([s["best_val_loss"] for s in runs])
            # diverged only catches nan/inf, so a finite loss no better than a constant counts as a failure too
            n_failed = sum(s["diverged"] or s["best_val_loss"] >= baseline for s in runs)
            # a seed that diverged before its first validation has best_val_loss = inf, which makes std nan
            with np.errstate(invalid="ignore"):
                std = losses.std()
            rows.append({"optimizer": name, "lr": lr, "mean_val_loss": losses.mean(), "std_val_loss": std, "n_failed": n_failed})
    return rows


def pick_best_lrs(rows):
    # lowest mean val loss among learning rates where every seed trained successfully
    best = {}
    for row in rows:
        if row["n_failed"] > 0:
            continue
        name = row["optimizer"]
        if name not in best or row["mean_val_loss"] < best[name]["mean_val_loss"]:
            best[name] = row
    return best


def print_report(rows, best, baseline):
    print(f"\nconstant baseline val MSE: {baseline:.4f}")
    for name in OPTIMIZERS:
        print(f"\n{name}")
        for row in rows:
            if row["optimizer"] != name:
                continue
            mark = "  <- best" if best.get(name) is row else ""
            print(f"  lr={row['lr']:<7g} val MSE {row['mean_val_loss']:.4g} ± {row['std_val_loss']:.2g}  failed {row['n_failed']}/{len(SEEDS)}{mark}")
        if name not in best:
            print("  every learning rate failed on at least one seed")
        elif best[name]["lr"] in (LRS[0], LRS[-1]):
            print("  best lr is at the edge of the grid, the real optimum may lie outside it")


def write_csv(path, rows):
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    d = prepare_data(seed=0, standardize=True)
    baseline = constant_baseline_loss(d)

    summaries = run_sweep(d)
    rows = summarize(summaries, baseline)
    best = pick_best_lrs(rows)

    print_report(rows, best, baseline)
    write_csv(SWEEP_DIR / "lr_table.csv", rows)
    write_csv(SWEEP_DIR / "best_lrs.csv", list(best.values()))
