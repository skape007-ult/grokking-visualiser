# Reproducibility check (M3 + Linux CPU)

2026-11-08. This replaces the interim note from 2026-11-05.

**Correction.** The interim note compared cloud seed 0 with 4,216 and concluded
that seed 0 grokked later on the Linux CPU. That was wrong. The README lists
grokking steps sorted, not by seed, and 4,216 is seed 2. On the M3, seed 0
groks at 9,868, and the cloud run's 10,065 is 2.0% later than that.

Grokking step throughout means the first recorded step with test accuracy
≥ 95%, as in `first_step` in `train/plot.py`.

## (a) Same-machine determinism on the M3

Software: Apple M3, CPU, torch 2.15.0, Python 3.14.9.

I ran the README's exact command:

`.venv/bin/python train/train.py --run baseline`

No other job from this repo ran during that run (16:10–16:19), but the machine was
not idle: an unrelated project's Python job and a browser were using CPU. An
earlier attempt, overlapped for about two minutes by three `less_data` runs, was
also identical on every array below, so machine load did not change the result in
these runs.

I compared its `outputs/baseline/` with `outputs/wd1.0__seed0/`, which came from
the weight-decay sweep (`--run baseline --seed 0 --set weight_decay=1.0 --tag
wd1.0__seed0`).

| array | shape | exactly equal | max abs diff |
|---|---|---|---|
| `run.json` `steps` | 213 | yes | 0.0 |
| `run.json` `train_loss` | 213 | yes | 0.0 |
| `run.json` `test_loss` | 213 | yes | 0.0 |
| `run.json` `train_acc` | 213 | yes | 0.0 |
| `run.json` `test_acc` | 213 | yes | 0.0 |
| `run.json` `embedding_steps` | 50 | yes | 0.0 |
| `embeddings.npy` (float32) | 50 × 97 × 128 | yes | 0.0 |

The two `embeddings.npy` files also have the same SHA-256. The `config` objects
in the two `run.json` files are equal, so `--set weight_decay=1.0` (the default
value) and `--tag` changed nothing that matters: `--tag` only picks the output
directory. The one field that differs is the wall-clock `runtime.seconds`
(552.6 vs 258.8), which is not a training result.

The bit-exact comparison covers seed 0 only. For all three seeds, the sweep's
retrained wd = 1.0 runs give the same grokking steps as the README: see the
"M3 (README)" and "M3 (rerun)" columns below. That is a match in grokking step,
not a bit-exact comparison.

## (b) Across hardware, per seed

| seed | M3 (README) | M3 (rerun) | cloud CPU (Fourier session) | cloud − M3 (steps) | cloud − M3 (%) |
|---|---|---|---|---|---|
| 0 | 9,868 | 9,868 | 10,065 | +197 | +2.0% |
| 1 | 5,785 | 5,785 | 5,452 | −333 | −5.8% |
| 2 | 4,216 | 4,216 | 3,973 | −243 | −5.8% |

The largest difference is 5.8%, so the grokking steps are within about 6%. The
seed order is the same on both machines: seed 0 is slowest and seed 2 fastest.
This is a 3-seed check, so it does not bound the difference in general.

Sources:

- **M3 (README):** per-seed curves in `site/data/runs/baseline.json`.
- **M3 (rerun):** `grok_step` for wd = 1.0 in [`wd_sweep.csv`](wd_sweep.csv),
  and for seed 0 also the run in (a).
- **Cloud CPU:** [`fourier_cloud.csv`](fourier_cloud.csv). The cloud thread
  count and full software stack are not recorded in that file.

Cloud seed 0 partial curve ([`fourier_cloud_timing.csv`](fourier_cloud_timing.csv)), with
the M3 value at the same recorded step:

| step | cloud test acc | M3 test acc |
|---|---|---|
| 3,599 | 7.3% | 6.9% |
| 5,785 | 11.5% | 11.3% |
| 8,939 | 54.7% | 58.4% |
| 9,868 | 93.9% | 97.2% |

The curves track each other closely. The cloud run is slightly behind near the
transition, which is consistent with its later 95% crossing. Partial curves are
available for seed 0 only.

## (c) Why the steps differ across hardware

Different CPUs and backends (BLAS and vector kernels) evaluate the same
floating-point sums in different orders, and floating-point addition is not associative, so results differ in the last bits;
thread count changes how the work is split and can do the same. Grokking
amplifies these tiny differences over thousands of AdamW steps, so the 95%
crossing can move even when the seed is the same. This is the expected
explanation and was not isolated here: no experiment varied the backend or the
thread count on its own.
