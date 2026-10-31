# CS360-Project1-MLP-with-optimizers

A NumPy MLP (8 → 16 → 16 → 1, ReLU) trained on California housing with five optimizers (SGD, SGD with momentum, Adam, RMSProp, AdaGrad), compared on speed, quality, reliability and cost per step.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

The first run downloads the California housing dataset into `data/`. On the python.org build of macOS Python this can fail with `SSL: CERTIFICATE_VERIFY_FAILED`; fix it once with `open "/Applications/Python 3.13/Install Certificates.command"`.

Optional checks: `pytest` runs the gradient, optimizer, metric and PyTorch-parity tests, and `python baselines.py` prints the mean / linear / sklearn MLP baselines.

## Running the experiment

Run everything from the repo root, in this order.

### 1. Learning-rate sweep, standardized features

```bash
python -m experiments.lr_sweep
```

Trains every optimizer at every learning rate in `LRS` (1e-4 … 1.0) with 5 seeds for 20 epochs: 5 × 9 × 5 = 225 runs. Large learning rates are expected to diverge; those runs are recorded, not errors. When it finishes it prints, per optimizer and lr, the mean ± std val MSE and how many seeds diverged or never beat a constant prediction, and marks the best lr.

Writes to `logs/sweep_std/`:

| File | Contents |
|---|---|
| `{run_id}.csv` | training rows of one run: `step, epoch, batch_loss, time, val_loss` (`val_loss` only on the last step of each epoch, `time` excludes validation) |
| `{run_id}.npz` | parameters at the best validation epoch |
| `summary_rows.csv` | one row per run: optimizer, lr, seed, diverged, best val loss, … |
| `lr_table.csv` | one row per optimizer × lr: mean/std val loss, diverged and no-better counts |
| `best_lrs.csv` | each optimizer's best lr: lowest mean val loss among lrs where every seed trained |

Settings live at the top of `experiments/lr_sweep.py`: `RUN_TRAINING = False` re-analyzes existing logs without training, and `FINAL_LRS` runs each optimizer only at one chosen lr.

### 2. Same sweep on raw features (for the feature-scale comparison)

Set `STANDARDIZED = False` in `experiments/lr_sweep.py` and run it again. Results go to `logs/sweep_raw/` in the same format. Set it back to `True` afterwards.

### 3. Metrics

```bash
python result.py
```

Reads `logs/sweep_std/` and prints the four metrics for each optimizer at its best lr, over its 5 seeds. It only prints; it writes no files. Example:

```
adam (lr=0.01)
  speed        2227 steps, 0.19 s to target (reached by 5/5 seeds)
  quality      mean best val MSE 0.3041
  reliability  seed range 0.2962–0.3156, diverged 0/45 runs, valid lrs 3/9
  cost         0.0811 ms/step, 2.00x sgd
```

| Metric | How it is computed | Read from |
|---|---|---|
| Speed | mean steps and seconds until val loss first drops to `SPEED_TARGET`, over the seeds that reached it; accurate to one epoch, since val loss is logged once per epoch | training rows |
| Quality | mean of each seed's best val loss, i.e. the lowest average val loss the optimizer reached | `summary_rows.csv` |
| Reliability | seed range (min–max of best val loss); diverged runs over the optimizer's whole sweep (always 0 at the best lr itself); valid lrs = lrs with no diverged seed whose mean val loss is within 10% of the optimizer's lowest | `summary_rows.csv` |
| Cost per step | median time per training step, and that divided by SGD's | training rows |

The valid-lr count needs the full sweep; after a `FINAL_LRS` run it is always 1/1.

### 4. Figures

```bash
python plots.py
```

Saves three PNGs to `figures/`:

- `val_loss_curves.png`: validation loss vs step and vs seconds. One line per optimizer (seed mean at its best lr), a min–max band, log y-axis, dashed lines for the linear-regression baseline and the speed target. Diverged runs are left out of the band and counted in the caption.
- `lr_sweep.png`: best validation loss vs learning rate (log x and y), one line per optimizer. A gap means a seed diverged at that lr.
- `feature_scale.png`: best validation loss per optimizer with standardized (filled) vs raw (hollow) features. Needs step 2; without it only the standardized points appear.

Each optimizer keeps the same color in every figure.

Where each figure's data comes from:

| Figure | Data source |
|---|---|
| `val_loss_curves.png` | `logs/sweep_std/best_lrs.csv` picks each optimizer's lr; `logs/sweep_std/summary_rows.csv` gives the `run_id`s of that lr's seeds and which diverged; each run's training rows `logs/sweep_std/{run_id}.csv` give the curve (`step`, `time`, `val_loss` at each epoch end) |
| `lr_sweep.png` | `logs/sweep_std/lr_table.csv` (`optimizer`, `lr`, `mean_val_loss`) |
| `feature_scale.png` | every `summary_rows.csv` under `logs/`, i.e. `logs/sweep_std/` and `logs/sweep_raw/` (`optimizer`, `lr`, `standardized`, `diverged`, `best_val_loss`) |
| linear baseline line | not from the logs: `plots.py` loads the dataset with `data.prepare_data` and fits least squares (`baselines.closed_form_linear`) to get its validation MSE |

## Where the data lives

| Path | What it is | Made by |
|---|---|---|
| `data/cal_housing_py3.pkz` | the raw California housing dataset (20,640 houses, 8 features, target = median house value in $100k), cached by sklearn | first call to `data.prepare_data` |
| `logs/sweep_std/` | training logs of the standardized-feature sweep (files listed in step 1) | `experiments/lr_sweep.py` |
| `logs/sweep_raw/` | the same for raw features | `experiments/lr_sweep.py` with `STANDARDIZED = False` |
| `figures/` | the three PNGs | `plots.py` |

`data/`, `logs/` and `.venv/` are in `.gitignore`, so none of these are committed. The dataset is split 70/15/15 into train/val/test with seed 0 (`data.split_indices`); standardization uses the training split's mean and std only.

## Before reporting results

`SPEED_TARGET` at the top of `plots.py` (used by both `plots.py` and `result.py`) is a placeholder of 0.35 val MSE. Set it to the project's target value before reading the speed metric.
