# CS360-Project1-MLP-with-optimizers

A NumPy MLP (8 → 16 → 16 → 1, ReLU) trained on California housing with five optimizers (SGD, SGD with momentum, Adam, RMSProp, AdaGrad), compared on speed, quality, reliability and cost per step.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Requires Python 3.12 or newer.

The first run downloads the California housing dataset into `data/`. 

Optional checks: `pytest` runs the gradient, optimizer, metric and PyTorch-parity tests, and `python baselines.py` prints the mean / linear / sklearn MLP baselines.

## Running the experiment

Run everything from the repo root, in this order. `lr_sweep.py` and `analyze.py` are configured by the constants at the top of each file.

Every training run uses the same network and batch size (48); the phases differ only in optimizers, learning rates, seeds and epochs:

| Phase | `lr_sweep.py` settings | Runs | Output |
|---|---|---|---|
| 0. Pilot | `PHASE = "pilot"`, SGD and AdaGrad, all 9 lrs, `SEEDS = [0]`, `N_EPOCHS = 200`, standardized | 18 | `logs/pilot_std/` |
| 1. Sweep | `PHASE = "sweep"`, all five optimizers, all 9 lrs, `SEEDS = [0, 1, 2, 3, 4]`, `N_EPOCHS = 260`, `FINAL_LRS = None`; once standardized, once raw | 225 + 225 | `logs/sweep_std/`, `logs/sweep_raw/` |
| 3. Final | `PHASE = "final"`, each optimizer at its winning lr only, `SEEDS = [5, 6, 7, 8, 9]`, `N_EPOCHS = 260`; once standardized, once raw | 25 + 15 | `logs/final_std/`, `logs/final_raw/` |

### 0. Pilot: choose the epoch budget

Run `python lr_sweep.py` with the pilot settings above, then `python analyze.py` with `MODE = "pilot"` and `STANDARDIZED = True`. For each optimizer's best lr it finds the epoch where the val loss first gets within 1% of its lowest, and prints the budget as twice the slowest of those. Our pilot: SGD stops improving at epoch 130 and AdaGrad at 98, so the budget is 2 × 130 = **260 epochs**, used by the sweep and the final runs.

### 1. Learning-rate sweep

```bash
python lr_sweep.py
```

Each optimizer is trained at every learning rate in `LRS` (1e-4 … 1.0) for every seed in `SEEDS`. Large learning rates are expected to diverge; those runs are recorded, not errors. It prints mean ± std val MSE per optimizer and lr, how many seeds diverged or never beat a constant prediction, and marks the best lr. Run it with `STANDARDIZED = True` and again with `False`. Each run folder holds:

| File | Contents |
|---|---|
| `{run_id}.csv` | training rows of one run: `step, epoch, batch_loss, time, val_loss` (`val_loss` only on the last step of each epoch, `time` excludes validation) |
| `{run_id}.npz` | parameters at the best validation epoch |
| `summary_rows.csv` | one row per run: optimizer, lr, seed, diverged, best val loss, … |
| `lr_table.csv` | one row per optimizer × lr: mean/std val loss, diverged and no-better counts |
| `best_lrs.csv` | each optimizer's best lr: lowest mean val loss among lrs where every seed trained |

With raw features SGD and Momentum diverge at every lr, so they have no best lr there.

### 2. Winners and working learning-rate range

Run `python analyze.py` with `MODE = "sweep"`, once with `STANDARDIZED = True` and once with `False`. It writes `winners.csv` into `logs/sweep_std/` and `logs/sweep_raw/`: each optimizer's best lr, the epoch where it stops improving, and its working range (lrs whose mean val loss is within 10% of the best) as `working_ratio` = largest / smallest working lr.

### 3. Final runs

Set `FINAL_LRS` to the `lr` column of the matching `winners.csv` and run `python lr_sweep.py` with the final settings above, once per feature scale. Fresh seeds (5–9) keep the final numbers independent of the seeds used to pick the lr. The learning rates we used:

| Optimizer | standardized (`logs/final_std/`) | raw (`logs/final_raw/`) |
|---|---|---|
| SGD | 0.01 | none: diverged at every lr |
| Momentum | 0.001 | none: diverged at every lr |
| Adam | 0.003 | 0.01 |
| RMSProp | 0.003 | 0.003 |
| AdaGrad | 0.3 | 0.03 |

Each run takes a few seconds (about 4 s for SGD and 8 s for Adam at 260 epochs), so a full final phase takes a few minutes.

### 4. Metrics

```bash
python results.py
```

Reads `logs/final_std/` and writes `results.csv` in the repo root, one row per optimizer (also printed):

| Metric | Columns | How it is computed | Read from |
|---|---|---|---|
| Quality | `mean_best_val_loss`, `sd_best_val_loss` | mean and sample SD (n−1) of `best_val_loss` across the 5 seeds | `logs/final_std/summary_rows.csv` |
| Speed | `steps_to_target`, `seconds_to_target`, `reached` | in each run, the first row with `val_loss` ≤ `SPEED_TARGET` (0.316); its step and time are averaged over the seeds that reached it, and `reached` counts them (e.g. 3/5). "not reached" when no seed does. Accurate to one epoch, since val loss is logged once per epoch | `logs/final_std/{run_id}.csv` |
| Reliability | `sd_best_val_loss`, `n_diverged`, `working_ratio` | the SD from Quality; diverged runs among the 5; the working-range ratio. Momentum is reported as 3.3× instead of 33×, because a seed diverged at lr 0.003, which lies inside that range | `summary_rows.csv`, `logs/sweep_std/winners.csv` |
| Cost per step | `median_step_ms`, `cost_vs_sgd` | per run, the median difference between consecutive `time` values, skipping epoch 1 (warm-up); averaged over seeds and divided by SGD's, so SGD = 1.0 | `logs/final_std/{run_id}.csv` |

### 5. Figures

```bash
python plots.py
```

Saves three PNGs to `figures/`:

- `val_loss_curves.png`: validation loss vs step and vs seconds for the final runs. One line per optimizer (seed mean at its winning lr), a min–max band, log y-axis, dashed lines for the linear-regression baseline and the speed target. Diverged runs are left out of the band and counted in the caption.
- `lr_sweep.png`: best validation loss vs learning rate (log x and y), one line per optimizer. A gap means a seed diverged at that lr.
- `feature_scale.png`: mean best validation loss per optimizer with standardized (filled) vs raw (hollow) features. SGD and Momentum have no raw point; they are labeled "diverged at every lr" in the plot and the caption.

Each optimizer keeps the same color in every figure.

Where each figure's data comes from:

| Figure | Data source |
|---|---|
| `val_loss_curves.png` | `logs/final_std/summary_rows.csv` gives each optimizer's lr, its `run_id`s and which diverged; each run's training rows `logs/final_std/{run_id}.csv` give the curve (`step`, `time`, `val_loss` at each epoch end) |
| `lr_sweep.png` | `logs/sweep_std/lr_table.csv` (`optimizer`, `lr`, `mean_val_loss`) |
| `feature_scale.png` | `logs/final_std/summary_rows.csv` and `logs/final_raw/summary_rows.csv` (`optimizer`, `best_val_loss`) |
| linear baseline line | not from the logs: `plots.py` loads the dataset with `data.prepare_data` and fits least squares (`baselines.closed_form_linear`) to get its validation MSE |

## Where the data lives

| Path | What it is | Made by |
|---|---|---|
| `data/cal_housing_py3.pkz` | the raw California housing dataset (20,640 houses, 8 features, target = median house value in $100k), cached by sklearn | first call to `data.prepare_data` |
| `logs/pilot_std/` | pilot runs used to pick the 260-epoch budget | `lr_sweep.py` with `PHASE = "pilot"` |
| `logs/sweep_std/`, `logs/sweep_raw/` | learning-rate sweep logs (files listed in step 1), each with its `winners.csv` | `lr_sweep.py` with `PHASE = "sweep"`, `analyze.py` |
| `logs/final_std/`, `logs/final_raw/` | final runs at each optimizer's winning lr | `lr_sweep.py` with `PHASE = "final"` |
| `results.csv` | the four metrics, one row per optimizer | `results.py` |
| `figures/` | the three PNGs | `plots.py` |

`data/`, `logs/` and `.venv/` are in `.gitignore`, so they are not committed: the logs exist only on the machine that ran the training, and have to be copied over to run `results.py` or `plots.py` elsewhere. The dataset is split 70/15/15 into train/val/test with seed 0 (`data.split_indices`); standardization uses the training split's mean and std only.
