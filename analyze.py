import csv
import numpy as np

from train import LOG_DIR

MODE = "pilot"  # "pilot" or "sweep"
STANDARDIZED = True


DIR = LOG_DIR / f"{MODE}_{"std" if STANDARDIZED else "raw"}"

STOP_MARGIN = 0.01   # stop epoch is the first epoch to reach val loss within 1% of lowest
TIE_MARGIN = 0.01   # other lrs within 1% of best
WORKING_MARGIN = 0.10    # val loss within 10% of lowest

def load_csv(path):
    with open(path, newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def val_loss_curve(run_id):
    curve = []
    for row in load_csv(DIR / f"{run_id}.csv"):
        if row["val_loss"] != "":   # only the last step of each epoch has a validation loss
            curve.append(float(row["val_loss"]))

    return np.array(curve)


def curve_mean(summaries, optimizer, lr):
    curves = []

    for summary in summaries:
        if summary["optimizer"] == optimizer and float(summary["lr"]) == lr:
            curves.append(val_loss_curve(summary["run_id"]))

    return np.mean(curves, axis=0)


def stop_epoch(curve):
    best = curve.min()
    for i in range(len(curve)):
        if curve[i] <= best * (1+STOP_MARGIN):
            return i+1

def main():
    summaries = load_csv(DIR / "summary_rows.csv")
    winners = load_csv(DIR / "best_lrs.csv")

    stops = {}
    for winner in winners:
        name = winner["optimizer"]
        lr = float(winner["lr"])
        curve = curve_mean(summaries, name, lr)
        stops[name] = stop_epoch(curve)

        print(f"{name}: lr={lr}, best val_loss {curve.min():.4f}, stops improving at {stops[name]} epoch out of {len(curve)}")

        best_epoch = int(np.argmin(curve)) + 1
        if best_epoch > 0.9 * len(curve):
            print("Warning: still improving near the end")

    if MODE == "pilot":
        slowest = 0
        for name in stops:
            if stops[name] > slowest:
                slowest = stops[name]

        print(f"Budget = 2 x {slowest} = {2 * slowest} epochs")
        return

        
        
        
    table = load_csv(DIR / "lr_table.csv")
    rows = []     

    for winner in winners:
        name = winner["optimizer"]
        w_lr = float(winner["lr"])
        w_loss = float(winner["mean_val_loss"])

        good = [] 
        failing_lr = []

        for row in table:
            if row["optimizer"] != name:
                continue
            if int(row["n_diverged"]) == 0 and int(row["n_no_better"]) == 0:
                good.append(row)
            else:
                failing_lr.append(float(row["lr"]))

        tie_lrs = []
        working = []

        for row in good:
            lr = float(row["lr"])
            loss = float(row["mean_val_loss"])

            if lr != w_lr and loss <= w_loss * (1+TIE_MARGIN):
                tie_lrs.append(lr)

            if loss <= w_loss * (1 + WORKING_MARGIN):
                working.append(lr)

        ratio = round(max(working)/min(working),1)

        rows.append({
            "optimizer": name,
            "lr": w_lr,
            "mean_val_loss": w_loss,
            "stop_epoch": stops[name],
            "working_lr_min": min(working),
            "working_lr_max": max(working),
            "working_ratio": ratio,
        })

        print(f"\n{name}: working lrs {working} ({ratio}x)")
        if tie_lrs:
            print(f"TIE: {tie_lrs} within 1% of {w_lr}; FAILED: {failing_lr}")


    with open(DIR / "winners.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {DIR/"winners.csv"}")

if __name__ == "__main__":
    main()

