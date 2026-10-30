# CS360-Project1-MLP-with-optimizers

## requirements
run
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

```
to install dependencies.

## plots.py

Plot:
	•	Validation loss vs step, and vs seconds: one line per optimizer (seed mean), a min–max band, a log y-axis, dashed lines for the linear baseline and the speed target, and the same color per optimizer in every figure. Leave diverged runs out of the band and count them in the caption.
	•	Learning-rate sweep: best validation loss vs learning rate (log x), one line per optimizer, from lr_table.csv.
	•	Feature scale: standardized vs raw best validation loss per optimizer.

using 

Run it after experiments/lr_sweep.py:
python plots.py            # set SPEED_TARGET at the top of plots.py first
It saves three PNGs to figures/: