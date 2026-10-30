import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import closed_form_linear
from data import prepare_data

LOG_DIR = Path(__file__).parent / "logs"
SWEEP_DIR = LOG_DIR / "lr_sweep"
FIG_DIR = Path(__file__).parent / "figures"

# val MSE an optimizer has to reach for the time-to-target comparison
SPEED_TARGET = 0.35

# same order as train.OPTIMIZERS; each optimizer keeps its color in every figure
OPTIMIZER_COLORS = {
    "sgd": "#2a78d6",
    "momentum": "#eb6834",
    "adam": "#1baf7a",
    "rmsprop": "#eda100",
    "adagrad": "#e87ba4",
}


def load_csv(path):
    with open(path, newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def load_summaries(paths):
    # a run logged in more than one summary file is kept once
    by_id = {}
    for path in paths:
        for row in load_csv(path):
            row["lr"] = float(row["lr"])
            row["standardized"] = row["standardized"] == "True"
            row["diverged"] = row["diverged"] == "True"
            row["best_val_loss"] = float(row["best_val_loss"])
            by_id[row["run_id"]] = row
    return list(by_id.values())


def load_val_curve(path):
    # val_loss is only filled in on the last step of each epoch
    rows = [r for r in load_csv(path) if r["val_loss"] != ""]
    step = np.array([int(r["step"]) for r in rows])
    seconds = np.array([float(r["time"]) for r in rows])
    val_loss = np.array([float(r["val_loss"]) for r in rows])
    return step, seconds, val_loss


def linear_baseline_loss():
    d = prepare_data(seed=0, standardize=True)
    return closed_form_linear(d)["val_rmse"] ** 2


def save(fig, path, caption):
    fig.text(0.01, 0.01, caption, fontsize=9, color="gray", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect_curves(summaries, best_lrs):
    """Per-epoch step, seconds and val loss of every finished seed at each optimizer's best lr"""
    curves = {}
    for name, lr in best_lrs.items():
        runs = [s for s in summaries if s["optimizer"] == name and s["standardized"] and np.isclose(s["lr"], lr)]
        # diverged runs stop early and blow up, so they stay out of the mean and the band
        finished = [load_val_curve(SWEEP_DIR / f"{r['run_id']}.csv") for r in runs if not r["diverged"]]
        curves[name] = {
            "lr": lr,
            "n_runs": len(runs),
            "n_diverged": len(runs) - len(finished),
            # every seed sees the same batches per epoch, so steps line up exactly
            "step": finished[0][0],
            # wall-clock differs per seed, so x is the mean time at which each epoch finished
            "seconds": np.mean([f[1] for f in finished], axis=0),
            "loss": np.array([f[2] for f in finished]),
        }
    return curves


def plot_loss_vs_step_and_time(curves, linear_baseline, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, x_key in zip(axes, ("step", "seconds")):
        for name, c in curves.items():
            color = OPTIMIZER_COLORS[name]
            ax.fill_between(c[x_key], c["loss"].min(axis=0), c["loss"].max(axis=0), color=color, alpha=0.2, linewidth=0)
            ax.plot(c[x_key], c["loss"].mean(axis=0), color=color, label=f"{name} (lr={c['lr']:g})")
        ax.axhline(linear_baseline, color="black", linestyle="--", label="linear baseline")
        ax.axhline(SPEED_TARGET, color="gray", linestyle=":", label="speed target")
        ax.set_yscale("log")
        ax.set_xlabel(x_key)
        ax.set_title(f"Validation loss vs {x_key}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation MSE")
    axes[1].legend()

    counts = ", ".join(f"{name} {c['n_diverged']}/{c['n_runs']}" for name, c in curves.items())
    save(fig, path, f"Line: seed mean at best lr. Band: min–max over seeds. Diverged runs left out: {counts}.")


def plot_lr_sweep(lr_rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, color in OPTIMIZER_COLORS.items():
        rows = sorted((r for r in lr_rows if r["optimizer"] == name), key=lambda r: float(r["lr"]))
        lr = [float(r["lr"]) for r in rows]
        loss = [float(r["mean_val_loss"]) for r in rows]
        # matplotlib skips inf, so a learning rate where a seed diverged leaves a gap
        ax.plot(lr, loss, "o-", color=color, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best validation MSE (seed mean)")
    ax.set_title("Learning-rate sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path, "Gaps: a seed diverged, so the mean is infinite.")


def best_by_scale(summaries):
    """Lowest seed-mean best val loss per (optimizer, standardized) over lrs where no seed diverged"""
    groups = {}
    for s in summaries:
        groups.setdefault((s["optimizer"], s["standardized"], s["lr"]), []).append(s)
    best = {}
    for (name, standardized, lr), runs in groups.items():
        if any(r["diverged"] for r in runs):
            continue
        loss = np.mean([r["best_val_loss"] for r in runs])
        best[(name, standardized)] = min(loss, best.get((name, standardized), np.inf))
    return best


def plot_feature_scale(summaries, linear_baseline, path):
    best = best_by_scale(summaries)
    names = list(OPTIMIZER_COLORS)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, name in enumerate(names):
        color = OPTIMIZER_COLORS[name]
        if (name, True) in best:
            ax.plot(i - 0.1, best[(name, True)], "o", color=color, markersize=9)
        if (name, False) in best:
            ax.plot(i + 0.1, best[(name, False)], "o", markerfacecolor="white", markeredgecolor=color, markersize=9, markeredgewidth=2)
    ax.plot([], [], "o", color="gray", label="standardized")
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="gray", markeredgewidth=2, label="raw")
    ax.axhline(linear_baseline, color="black", linestyle="--", label="linear baseline")
    ax.set_xticks(range(len(names)), names)
    ax.set_yscale("log")
    ax.set_ylabel("best validation MSE (seed mean)")
    ax.set_title("Feature scale: standardized vs raw")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    save(fig, path, "Each point: best lr where no seed diverged.")


def main():
    linear_baseline = linear_baseline_loss()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    best_lrs = {r["optimizer"]: float(r["lr"]) for r in load_csv(SWEEP_DIR / "best_lrs.csv")}
    curves = collect_curves(load_summaries([SWEEP_DIR / "summary_rows.csv"]), best_lrs)
    plot_loss_vs_step_and_time(curves, linear_baseline, FIG_DIR / "val_loss_curves.png")

    plot_lr_sweep(load_csv(SWEEP_DIR / "lr_table.csv"), FIG_DIR / "lr_sweep.png")

    # raw-feature runs can live in their own log folder, so read every summary under logs/
    plot_feature_scale(load_summaries(sorted(LOG_DIR.rglob("summary_rows.csv"))), linear_baseline, FIG_DIR / "feature_scale.png")

    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
