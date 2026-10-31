import csv

import numpy as np

from data import prepare_data
from train import LOG_DIR, OPTIMIZERS, training_run



PHASE = "final"

STANDARDIZED = False
SWEEP_DIR = LOG_DIR / f"{PHASE}_{"std" if STANDARDIZED else "raw"}"

#OPTIMIZER_NAMES = list(OPTIMIZERS)          # full sweep: all five
OPTIMIZER_NAMES = ["adam", "rmsprop", "adagrad"]      # pilot

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
# finals: each optimizer runs only at its own winning learning rate instead of every value in LRS
#FINAL_LRS = None
FINAL_LRS = {"adam": 0.01, "rmsprop": 0.003, "adagrad": 0.03}

SEEDS = [5,6,7,8,9]
N_EPOCHS = 260
# False: skip training and only re-analyze the summary_rows.csv already in SWEEP_DIR
RUN_TRAINING = True


def lrs_for(name):
    if FINAL_LRS is None:
        return LRS
    return [FINAL_LRS[name]]


def constant_baseline_loss(d):
    # the best any constant prediction can do on val is var(y_val), so a network that died and outputs a constant never gets below it
    return np.var(d["y_val"])


def run_sweep(d):
    # training_run appends to summary_rows.csv, so clear it first or a rerun would duplicate every row
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "summary_rows.csv").unlink(missing_ok=True)

    for name in OPTIMIZER_NAMES:
        print(f"running {name}")
        for lr in lrs_for(name):
            for seed in SEEDS:
                # large learning rates are expected to overflow, training_run already records that as diverged
                with np.errstate(all="ignore"):
                    training_run(d, name, lr, seed, N_EPOCHS, standardized=STANDARDIZED, log_dir=SWEEP_DIR)


def load_summaries():
    # csv gives everything back as text, so convert the columns the analysis compares
    with open(SWEEP_DIR / "summary_rows.csv") as f:
        summaries = list(csv.DictReader(f))
    for s in summaries:
        s["lr"] = float(s["lr"])
        s["best_val_loss"] = float(s["best_val_loss"])
        s["diverged"] = s["diverged"] == "True"
    return summaries


def summarize(summaries, baseline):
    rows = []
    for name in OPTIMIZER_NAMES:
        for lr in lrs_for(name):
            runs = [s for s in summaries if s["optimizer"] == name and s["lr"] == lr]
            # check in case sweep was interrrupted and skipped or duplicated runs
            if len(runs) != len(SEEDS):
                print(f"{name} lr={lr} has {len(runs)} of {len(SEEDS)} runs")
                continue
            losses = np.array([s["best_val_loss"] for s in runs])
            # diverged: the loss became nan/inf
            n_diverged = sum(s["diverged"] for s in runs)
            # no better: the loss stayed finite but never beat a constant, e.g. dead ReLUs or a learning rate too small to learn in time
            n_no_better = sum(not s["diverged"] and s["best_val_loss"] >= baseline for s in runs)
            # a seed that diverged before its first validation has best_val_loss = inf, which makes std nan
            with np.errstate(invalid="ignore"):
                std = losses.std()
            rows.append({"optimizer": name, "lr": lr, "mean_val_loss": losses.mean(), "std_val_loss": std, "n_diverged": n_diverged, "n_no_better": n_no_better})
    return rows


def pick_best_lrs(rows):
    # lowest mean val loss among learning rates where every seed trained successfully
    best = {}
    for row in rows:
        if row["n_diverged"] + row["n_no_better"] > 0:
            continue
        name = row["optimizer"]
        if name not in best or row["mean_val_loss"] < best[name]["mean_val_loss"]:
            best[name] = row
    return best


def print_report(rows, best, baseline):
    print(f"\nconstant baseline val MSE: {baseline:.4f}")
    for name in OPTIMIZER_NAMES:
        print(f"\n{name}")
        for row in rows:
            if row["optimizer"] != name:
                continue
            mark = "  <- best" if best.get(name) is row else ""
            print(f"  lr={row['lr']:<7g} val MSE {row['mean_val_loss']:.4g} ± {row['std_val_loss']:.2g}  diverged {row['n_diverged']}/{len(SEEDS)}  no better {row['n_no_better']}/{len(SEEDS)}{mark}")
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
    d = prepare_data(seed=0, standardize=STANDARDIZED)
    baseline = constant_baseline_loss(d)

    if RUN_TRAINING:
        run_sweep(d)
    summaries = load_summaries()
    rows = summarize(summaries, baseline)
    best = pick_best_lrs(rows)

    print_report(rows, best, baseline)
    write_csv(SWEEP_DIR / "lr_table.csv", rows)
    write_csv(SWEEP_DIR / "best_lrs.csv", list(best.values()))
