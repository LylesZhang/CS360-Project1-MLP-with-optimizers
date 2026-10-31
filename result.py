import numpy as np

from plots import SWEEP_DIR, SPEED_TARGET, load_csv, load_summaries

# an lr counts as valid when its seed-mean val loss is within 10% of the optimizer's lowest
VALID_LR_TOLERANCE = 0.10


def load_run(path):
    """(step, seconds) when val loss first reaches SPEED_TARGET or None, and the duration of every step"""
    rows = load_csv(path)
    # val loss is only logged at the end of each epoch, so the hit is accurate to one epoch
    hit = next(((int(r["step"]), float(r["time"])) for r in rows
                if r["val_loss"] != "" and float(r["val_loss"]) <= SPEED_TARGET), None)
    # time is cumulative and already excludes validation, so its differences are the cost of each step
    step_times = np.diff([float(r["time"]) for r in rows], prepend=0.0)
    return hit, step_times


def count_valid_lrs(sweep):
    by_lr = {}
    for r in sweep:
        by_lr.setdefault(r["lr"], []).append(r)
    # an lr where a seed diverged is never valid
    means = [np.mean([r["best_val_loss"] for r in runs]) for runs in by_lr.values() if not any(r["diverged"] for r in runs)]
    n_valid = sum(m <= min(means) * (1 + VALID_LR_TOLERANCE) for m in means) if means else 0
    return n_valid, len(by_lr)


def evaluate(name, lr, summaries):
    sweep = [s for s in summaries if s["optimizer"] == name]
    runs = [s for s in sweep if np.isclose(s["lr"], lr) and not s["diverged"]]
    loaded = [load_run(SWEEP_DIR / f"{r['run_id']}.csv") for r in runs]
    hits = [hit for hit, _ in loaded if hit is not None]
    best = [r["best_val_loss"] for r in runs]
    return {
        "lr": lr,
        "n_seeds": len(runs),
        "n_reached": len(hits),
        "steps": np.mean([h[0] for h in hits]) if hits else np.nan,
        "seconds": np.mean([h[1] for h in hits]) if hits else np.nan,
        "quality": np.mean(best),
        "min": np.min(best),
        "max": np.max(best),
        "n_diverged": sum(s["diverged"] for s in sweep),
        "n_runs": len(sweep),
        "valid_lrs": count_valid_lrs(sweep),
        "ms_per_step": 1000 * np.median(np.concatenate([t for _, t in loaded])),
    }


def main():
    summaries = load_summaries([SWEEP_DIR / "summary_rows.csv"])
    best_lrs = {r["optimizer"]: float(r["lr"]) for r in load_csv(SWEEP_DIR / "best_lrs.csv")}
    results = {name: evaluate(name, lr, summaries) for name, lr in best_lrs.items()}
    sgd_ms = results["sgd"]["ms_per_step"] if "sgd" in results else np.nan

    print(f"each optimizer at its best lr, over its seeds; speed target: val MSE <= {SPEED_TARGET}")
    for name, r in results.items():
        n_valid, n_lrs = r["valid_lrs"]
        print(f"\n{name} (lr={r['lr']:g})")
        if r["n_reached"]:
            print(f"  speed        {r['steps']:.0f} steps, {r['seconds']:.2f} s to target (reached by {r['n_reached']}/{r['n_seeds']} seeds)")
        else:
            print(f"  speed        never reached the target (0/{r['n_seeds']} seeds)")
        print(f"  quality      mean best val MSE {r['quality']:.4f}")
        print(f"  reliability  seed range {r['min']:.4f}–{r['max']:.4f}, diverged {r['n_diverged']}/{r['n_runs']} runs, valid lrs {n_valid}/{n_lrs}")
        print(f"  cost         {r['ms_per_step']:.4f} ms/step, {r['ms_per_step'] / sgd_ms:.2f}x sgd")


if __name__ == "__main__":
    main()
