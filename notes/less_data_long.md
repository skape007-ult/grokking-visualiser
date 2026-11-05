# less_data at a long horizon (60,000 steps)

**Question:** does `less_data` (15% of the 97×97 sums, weight decay 1.0), which is at chance after the
default 25,000 steps, grok if trained much longer?

**Answer within this horizon: no.** All three seeds memorise by step ~110 and stay at chance on unseen
sums through step 60,000.

## What was run

```
OMP_NUM_THREADS=1 python train/train.py --run less_data --seed {0,1,2} --set steps=60000 --tag long60k_seed{0,1,2}
```

- **Steps: 60,000, not the 100,000 asked for.** Timing on this 4-core CPU container: one run alone took
  81 s for 2,000 steps (40.5 ms/step, so 100k ≈ 68 min). Three in parallel took 195 s for 2,000 steps
  (~97 ms/step each), so 100k would have needed ~2h40m and missed the 02:30 deadline. 60k was the
  largest round number that fitted. Actual wall time: 4,791–5,216 s per run (80–87 min), all 3 in parallel.
- Every other setting is the repo's `less_data` config, unchanged. No code or config edits.
- Metrics come from the standard `run.json` logging: 200 log-spaced evaluation points per run (the adaptive
  dense recording never triggered, since test accuracy never moved).
- Definitions from `notes/log.md`: memorisation = first recorded step with train acc ≥ 99%;
  grokking = first recorded step with test acc ≥ 95%.

## Results

| seed | final train acc | final test acc | max test acc (any step) | train ≥ 99% at | grokking step |
|---|---|---|---|---|---|
| 0 | 100% | 0.74% | 1.04% | 107 | did not grok within 60,000 steps |
| 1 | 100% | 0.33% | 1.15% | 107 | did not grok within 60,000 steps |
| 2 | 100% | 0.84% | 1.61% | 112 | did not grok within 60,000 steps |

Chance is 1/97 ≈ 1.03%. Final test loss ≈ 39–40 nats (vs ln 97 ≈ 4.57 for a uniform guess), so the
model is confidently wrong on unseen sums, not uncertain.

Files: `less_data_long.png` (train dashed / test solid, log-x, chance dotted),
`less_data_long.csv` (~53 log-spaced points per seed: step, seed, train_acc, test_acc).

## Interpretation

Extending training from 25k to 60k steps changes nothing for `less_data`: every seed memorises the
15% training set in ~110 steps and test accuracy then sits flat at or slightly below chance for the
remaining ~59,900 steps, with no upward drift late in training. For comparison, the 30% baseline
groks at 4k–10k steps, i.e. 30–75× after memorisation; here we have waited ~550× past memorisation
without any sign of a transition. The test loss stays near 40 nats rather than falling, which suggests
weight decay is not slowly pulling the network toward a generalising solution at this data fraction —
the memorising solution looks stable. This is consistent with a data-fraction threshold somewhere
between 15% and 30% below which this setup does not grok in practical time, but these runs cannot
rule out a much later transition.

## Limitations

- **60k, not 100k, steps** (CPU time limit). "Did not grok" means not within 60,000 steps; a transition
  later than that is not excluded.
- Three seeds, one learning rate (1e-3) and one weight decay (1.0); grokking at low data fractions is
  known to be sensitive to both, and higher weight decay might change the outcome.
- Evaluation is on a log-spaced grid (200 points), so late-run resolution is coarse (~1.5k steps
  between points near 60k). A brief excursion between points would be missed, but a sustained climb
  to 95% would not.
- Runs used `OMP_NUM_THREADS=1` with 3 processes sharing 4 cores; this should affect speed only, but
  results may not be bit-identical to the repo's default-thread runs.
