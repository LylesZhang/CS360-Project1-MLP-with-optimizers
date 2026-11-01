# CS360 Project 1: Comparing Different Gradient Optimizers on 2-hidden-layer MLP

Requirements: Python 3.12 or newer

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

```

First run downloads scikit-learn California Housing dataset into data/.

Check the code runs
```bash
python -m pytest
```
All 58 tests should pass.


## Fixed Settings

1. Model: 2 hidden layer MLP [8, 16, 16, 1], ReLU, 433 parameters, float32
2. Loss: MSE
3. Batch size: 48
4. Split: 70/15/15 = 14,448/3,096/3,096 rows training/validation/test, fixed by seed 0 
5. Hyperparameters: only learning rate changes, PyTorch defaults for rest
6. Learning rates sweeped: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1
7. Seeds: 0-4 for sweeps, 5-9 for finals
8. Divergence: Stop run if batch or validation loss is NaN or inf
9. Failed run: Diverged or best validation loss is never better than a constant (var(y_val))
10. Final weights: Best validation run weights, saved as .npz


## Reproducing the Experiment
Run every step in order, one at a time. Do not interrupt a sweep.

For each phase, edit the following constants in lr_sweep.py and analyze.py (below), be sure to save, then run.

| Phase | `PHASE` | `STANDARDIZED` | `OPTIMIZER_NAMES` | `FINAL_LRS` | `SEEDS` | `N_EPOCHS` |
| --- | --- | --- | --- | --- | --- | --- |
| A. Pilot | `"pilot"` | `True` | `["sgd", "adagrad"]` | `None` | `[0]` | `200` |
| B. Standardized sweep | `"sweep"` | `True` | `list(OPTIMIZERS)` | `None` | `[0, 1, 2, 3, 4]` | `260` |
| C. Raw sweep | `"sweep"` | `False` | `list(OPTIMIZERS)` | `None` | `[0, 1, 2, 3, 4]` | `260` |
| D. Standardized finals | `"final"` | `True` | `list(OPTIMIZERS)` | standardized dict (below) | `[5, 6, 7, 8, 9]` | `260` |
| E. Raw finals | `"final"` | `False` | `["adam", "rmsprop", "adagrad"]` | raw dict (below) | `[5, 6, 7, 8, 9]` | `260` |

Two constants:
- LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
- RUN_TRAINING = True

### A. Pilot
Edit constants in lr_sweep.py (refer to table)
In analyze.py, set MODE to "pilot", and STANDARDIZED to True.

```bash
python lr_sweep.py
python analyze.py
```
analyze.py prints each winner's stop epoch, the first epoch whose validation loss is within 1% of the best achieved (within 200 epochs), and the budget (B) is set to twice the later of the two. We got SGD at epoch 130 and AdaGrad at 98, so B = 260.

### B. Standardized sweep
Edit constants in lr_sweep.py (refer to table)
In analyze.py, set MODE to "sweep", and STANDARDIZED to True.

```bash
python lr_sweep.py
python analyze.py
```
lr_sweep.py writes lr_table.csv (every optimizer × learning rate) and best_lrs.csv (the winners). analyze.py writes winners.csv (stop epoch and working range) and prints a warning for any tieing lrs.

### C. Raw sweep
Edit constants in lr_sweep.py (refer to table)
In analyze.py, set MODE to "sweep", and STANDARDIZED to False.

```bash
python lr_sweep.py
python analyze.py
```
SGD and Momentum diverge at every learning rate here, so only Adam, RMSProp and AdaGrad have raw winners.

### D. Standardized finals
Edit constants in lr_sweep.py (refer to table)
Set FINAL_LRS to winner from B (ours: FINAL_LRS = {"sgd": 0.01, "momentum": 0.001, "adam": 0.003, "rmsprop": 0.003, "adagrad": 0.3})

```bash
python lr_sweep.py
```

### E. Raw finals
Edit constants in lr_sweep.py (refer to table)
Set FINAL_LRS to winner from C (ours: FINAL_LRS = {"adam": 0.01, "rmsprop": 0.003, "adagrad": 0.03})

```bash
python lr_sweep.py
```

### F. Baselines, test set, results, figures
```bash
python baselines.py
python final_test.py
python results.py
python plots.py
```

## Notes (see before reproduction)
1. In step B & C, analyze.py will warn that most winners are still improving. This is expected. SGD and Momentum were improving by 1-2% near the end and others less, so instead of extending runs to hours we gave them the same 260 epoch budget.
2. Momentum's working range is 3.3x, not the 33x in winners.csv. This is because a seed diverged at lr 0.003, which is between 0.001 and 0.01, so the actual working range is from 0.0003 to 0.001.
