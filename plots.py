import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import closed_form_linear
from data import prepare_data

LOG_DIR = Path(__file__).parent / "logs"
SWEEP_DIR = LOG_DIR / "sweep_std"
# final runs: each optimizer at its winning lr, with standardized and raw features
FINAL_STD_DIR = LOG_DIR / "final_std"
FINAL_RAW_DIR = LOG_DIR / "final_raw"
FIG_DIR = Path(__file__).parent / "figures"

# val MSE an optimizer has to reach for the time-to-target comparison
SPEED_TARGET = 0.316

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


def load_summaries(path):
    rows = load_csv(path)
    for row in rows:
        row["lr"] = float(row["lr"])
        row["standardized"] = row["standardized"] == "True"
        row["diverged"] = row["diverged"] == "True"
        row["best_val_loss"] = float(row["best_val_loss"])
    return rows


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


def collect_curves(summaries):
    """Per-epoch step, seconds and val loss of every finished seed of each optimizer's final runs"""
    curves = {}
    for name in OPTIMIZER_COLORS:
        # the final runs use one lr per optimizer, its winner from the sweep
        runs = [s for s in summaries if s["optimizer"] == name]
        if not runs:
            continue
        # diverged runs stop early and blow up, so they stay out of the mean and the band
        finished = [load_val_curve(FINAL_STD_DIR / f"{r['run_id']}.csv") for r in runs if not r["diverged"]]
        curves[name] = {
            "lr": runs[0]["lr"],
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
    save(fig, path, f"Line: seed mean at the winning lr. Band: min–max over seeds. Diverged runs left out: {counts}.")


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


def mean_best_loss(summaries):
    by_name = {}
    for s in summaries:
        by_name.setdefault(s["optimizer"], []).append(s["best_val_loss"])
    return {name: np.mean(losses) for name, losses in by_name.items()}


def plot_feature_scale(std_summaries, raw_summaries, linear_baseline, path):
    std, raw = mean_best_loss(std_summaries), mean_best_loss(raw_summaries)
    names = list(OPTIMIZER_COLORS)
    # an optimizer has no final raw runs because it diverged at every lr of the raw sweep
    no_raw = [name for name in names if name in std and name not in raw]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, name in enumerate(names):
        color = OPTIMIZER_COLORS[name]
        if name in std:
            ax.plot(i - 0.1, std[name], "o", color=color, markersize=9)
        if name in raw:
            ax.plot(i + 0.1, raw[name], "o", markerfacecolor="white", markeredgecolor=color, markersize=9, markeredgewidth=2)
        elif name in no_raw:
            ax.annotate("raw: diverged\nat every lr", (i, std[name]), xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=8, color="gray")
    ax.plot([], [], "o", color="gray", label="standardized")
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="gray", markeredgewidth=2, label="raw")
    ax.axhline(linear_baseline, color="black", linestyle="--", label="linear baseline")
    ax.set_xticks(range(len(names)), names)
    ax.set_yscale("log")
    ax.set_ylabel("best validation MSE (seed mean)")
    ax.set_title("Feature scale: standardized vs raw")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    caption = "Each point: mean best val loss over the final runs at the winning lr."
    if no_raw:
        caption += f" No raw point for {', '.join(no_raw)}: diverged at every lr with raw features."
    save(fig, path, caption)


def main():
    linear_baseline = linear_baseline_loss()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    std_summaries = load_summaries(FINAL_STD_DIR / "summary_rows.csv")
    plot_loss_vs_step_and_time(collect_curves(std_summaries), linear_baseline, FIG_DIR / "val_loss_curves.png")

    plot_lr_sweep(load_csv(SWEEP_DIR / "lr_table.csv"), FIG_DIR / "lr_sweep.png")

    raw_summaries = load_summaries(FINAL_RAW_DIR / "summary_rows.csv")
    plot_feature_scale(std_summaries, raw_summaries, linear_baseline, FIG_DIR / "feature_scale.png")

    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
