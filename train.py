import time, csv

import numpy as np

from batch import make_batches
from model import MLP
from metrics import mse
from data import prepare_data
from optimizers import SGD, SGD_momentum, Adam, AdaGrad, RMSProp

from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
RUN_COLUMNS = ["step", "epoch", "batch_loss", "time", "val_loss"]
SUMMARY_COLUMNS = ["run_id", "optimizer", "lr", "seed", "standardized", "batch_size", "diverged", "epochs_run", "best_val_loss", "best_epoch"]

LAYER_SIZES = [8, 16, 16, 1]
BATCH_SIZE = 48

OPTIMIZERS = {
    "sgd": SGD,
    "momentum": SGD_momentum,
    "adam": Adam,
    "rmsprop": RMSProp,
    "adagrad": AdaGrad,
}

def training_run(d, optimizer_name, lr, seed, n_epochs, standardized=True, log_dir=LOG_DIR):
    model = MLP(LAYER_SIZES, dtype=np.float32)

    init_rng = np.random.default_rng(seed)
    batch_rng = np.random.default_rng(seed + 10)

    params = model.init(init_rng)

    optimizer = OPTIMIZERS[optimizer_name](lr)

    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]

    rows = []
    step = 0
    diverged = False
    best_val_loss = np.inf
    best_epoch = 0
    best_params = None

    start = time.perf_counter()

    for epoch in range(1, n_epochs+1):
        batches = make_batches(X_train, y_train, BATCH_SIZE, batch_rng)

        for Xb, yb in batches:
            loss, grads = model.loss_and_grads(params, Xb, yb)
            optimizer.step(params, grads)
            step += 1
            rows.append({"step":step, "epoch":epoch, "batch_loss":loss, "time": time.perf_counter() - start, "val_loss":""})
            if not np.isfinite(loss):
                diverged = True
                break

        if diverged:
            break

        pause = time.perf_counter()
        val_loss = mse(model.predict(params, X_val), y_val)
        rows[-1]["val_loss"] = val_loss

        if not np.isfinite(val_loss):
            diverged = True
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = {}
            for name,array in params.items():
                best_params[name] = array.copy()

        start += time.perf_counter() - pause

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id(optimizer_name, lr, seed, standardized)  

    with open(log_dir/f"{run_id}.csv","w", newline="") as csvfile:
        writer =csv.DictWriter(csvfile, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_id": run_id,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed":seed,
        "standardized":standardized,
        "batch_size": BATCH_SIZE,
        "diverged":diverged,
        "epochs_run":epoch,
        "best_val_loss":best_val_loss,
        "best_epoch":best_epoch,
    }

    summary_path = log_dir/"summary_rows.csv"
    write_header = not summary_path.exists()

    with open(summary_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    if best_params is not None:
        np.savez(log_dir/f"{run_id}.npz", **best_params)

    return summary, best_params

def make_run_id(optimizer_name, lr, seed, standardized):
    if standardized:
        data_state = "standardized"
    else:
        data_state = "raw"
    return f"{data_state}_{optimizer_name}_lr{lr}_seed{seed}"

if __name__ == "__main__":
    d = prepare_data(seed=0, standardize=True)

    n_epochs = 20
    start_time = time.perf_counter()
    summary, best_params = training_run(d, "adam", 1e-3, seed=0, n_epochs=n_epochs, log_dir=LOG_DIR/"timing")
    run_duration = time.perf_counter() - start_time

    print(f"{run_duration}s for {n_epochs} epochs")
    print(summary)