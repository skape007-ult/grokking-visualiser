# less_data at a long horizon: 100,000 steps on the M3

2026-11-08. Branch `exp/less-data-100k`. Data: [`less_data_long.csv`](less_data_long.csv). Plot: [`less_data_long.png`](less_data_long.png). Script: [`less_data_long.py`](less_data_long.py).

**Question:** does `less_data` (15% of the 97×97 sums, weight decay 1.0), which is at chance after the default 25,000 steps, grok if trained for 100,000 steps?

**Answer within this horizon: no.** All three seeds memorised by step 109 and none reached 95% test accuracy by step 100,000. Test accuracy stayed near chance (1/97 ≈ 1.03%).

## What was run

```
.venv/bin/python train/train.py --run less_data --seed S --set steps=100000 --tag long100k__seedS
```

- **Runs:** seeds 0, 1 and 2, in parallel on the Apple M3.
- **Device and threads:** CPU (the config default, as in the README), default thread count, torch 2.15.0, float64 loss. No code or config defaults were changed.
- **Time:** the first 1,960 steps took 39–40 s (about 20 ms/step), which estimated 33–35 min for 100k. The actual times were 1,697–1,700 s (about 28 min) per run. Another session's `baseline` training job shared the machine for the first part of these runs, so these times are not a clean benchmark.
- **Definitions:** memorisation is the first *recorded* step with train accuracy ≥ 99%. Grokking is the first *recorded* step with test accuracy ≥ 95% (`first_step` in `train/plot.py`).

**Recording density at 100k:** adequate, with defaults unchanged. Each run has 200 log-spaced recorded points and 40 embedding snapshots, about 48 points per decade above step 1,000, with neighbouring points at most 4.9% apart. The last gap is 4,627 steps. The adaptive dense recording never triggered, because test accuracy never moved fast enough. The test-accuracy curve is smooth on the log-x axis.

## Results (M3, 100,000 steps)

| seed | memorisation step (train ≥ 99%) | max test acc | final test acc | final test loss | vs ln(97) ≈ 4.57 | grokking step (test ≥ 95%) |
|---|---|---|---|---|---|---|
| 0 | 109 | 1.04% (step 1) | 0.71% | 41.18 | 9.0× | did not grok within 100,000 steps |
| 1 | 109 | 1.15% (step 4) | 0.46% | 40.33 | 8.8× | did not grok within 100,000 steps |
| 2 | 109 | 1.56% (step 71) | 0.90% | 38.94 | 8.5× | did not grok within 100,000 steps |

The maximum test accuracies all occur before memorisation. From memorisation onward the maxima are 0.79%, 0.50% and 1.51% for seeds 0, 1 and 2. Over steps 50,000 to 100,000 test accuracy ranged over 0.59–0.71%, 0.34–0.50% and 0.78–1.03%.

## Test-loss trend

Test loss at the last recorded step at or before each checkpoint (recorded steps 963, 9,812, 24,139, 49,131, 71,774 and 100,000):

| seed | 1,000 | 10,000 | 25,000 | 50,000 | 75,000 | 100,000 | peak (step) |
|---|---|---|---|---|---|---|---|
| 0 | 36.29 | 42.13 | 41.77 | 41.32 | 41.26 | 41.18 | 42.22 (3,628) |
| 1 | 35.84 | 40.46 | 40.54 | 40.38 | 40.42 | 40.33 | 41.11 (1,700) |
| 2 | 35.52 | 40.60 | 40.47 | 39.85 | 39.23 | 38.94 | 41.27 (2,369) |

The trend rule was fixed before the results were seen: an ordinary least-squares line of test loss against step over a window, "flat" if the fitted change across the window is under 1% of the window's mean test loss, otherwise "increasing" or "falling".

| seed | window | points | fitted change (nats) | % of window mean | trend |
|---|---|---|---|---|---|
| 0 | 50,000–100,000 | 15 | −0.040 | −0.10% | flat |
| 0 | 80,000–100,000 | 5 | −0.175 | −0.42% | flat |
| 1 | 50,000–100,000 | 15 | −0.054 | −0.13% | flat |
| 1 | 80,000–100,000 | 5 | −0.084 | −0.21% | flat |
| 2 | 50,000–100,000 | 15 | −0.832 | −2.12% | falling |
| 2 | 80,000–100,000 | 5 | −0.125 | −0.32% | flat |

No seed's test loss is increasing at the end. Seeds 0 and 1 are flat in both windows. Seed 2 is falling over the second half of the run and flat over the last 20,000 steps, where the window has only 5 points. Every fitted change is negative, and every run ended below its peak, by 1.04, 0.78 and 2.33 nats. Final test loss is still 8.5–9.0× ln(97), and seed 2's test accuracy over the same period stayed within 0.78–1.03%. These runs neither show nor exclude a later transition.

## Secondary: 60,000-step cloud CPU run (PR #2)

Earlier run on a 4-core cloud CPU, `OMP_NUM_THREADS=1`, tags `long60k_seed{0,1,2}`. Its curves are in [`less_data_long_cloud60k.csv`](less_data_long_cloud60k.csv), a verbatim copy of the CSV from PR #2 (53 points per seed, thinned from 200, no loss columns). The memorisation, final-accuracy and maximum figures below are those reported in the PR #2 note, from the full 200-point logs.

| seed | train ≥ 99% at | max test acc | final test acc |
|---|---|---|---|
| 0 | 107 | 1.04% | 0.74% |
| 1 | 107 | 1.15% | 0.33% |
| 2 | 112 | 1.61% | 0.84% |

PR #2 reports final test loss "≈ 39–40 nats" and did not grok within 60,000 steps.

**Do M3 and cloud agree over the first 60,000 steps?** On accuracy, yes.

- Interpolating the M3 curves (in log step) onto the cloud's 53 steps, test accuracy differs by at most 0.12, 0.16 and 0.18 percentage points for seeds 0, 1 and 2. Train accuracy differs by at most 0.71, 0.32 and 0.71 points, all before step 1,000, and not at all after it.
- The runs are not bit-identical. At shared step 82, seed 2's test accuracy is 1.28% on the M3 and 1.23% on the cloud CPU.
- Memorisation at 109/109/109 (M3) against 107/107/112 (cloud) reflects different recording grids, since the M3 grid steps through 104 and 109 and the cloud grid through 107 and 112. The values are compatible with crossing at about the same step.
- **Test loss can only be compared loosely.** Per-seed cloud losses were not committed. At M3 step 59,383 test loss is 41.26, 40.40 and 39.45, so seed 0 is about 1.3 nats above the cloud's reported range and the others are within or near it.

## Limitations

- "Did not grok" means not within 100,000 steps. A later transition is not excluded.
- Three seeds, one learning rate (1e-3), one weight decay (1.0).
- Evaluation is on a log-spaced grid, so late-run points are about 5% apart. A brief excursion between points would be missed.
- The trend windows have few points (15 and 5), and the 1% flat threshold is a chosen convention.
