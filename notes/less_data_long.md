# less_data at a long horizon — PARTIAL (runs in progress)

Status: interim snapshot at step 15,769 of 60,000 (~26%). Final numbers will replace this.

Runs: `python train/train.py --run less_data --seed {0,1,2} --set steps=60000 --tag long60k_seed{0,1,2}`
(100k did not fit the deadline; see final version for timing.)

| seed | step | train acc | test acc | train ≥99% at | test ≥95% at |
|---|---|---|---|---|---|
| 0 | 15,769 | 100% | 0.6% | 123 | not yet |
| 1 | 15,769 | 100% | 0.4% | 123 | not yet |
| 2 | 15,769 | 100% | 1.0% | 123 | not yet |

Interim curves (png/csv) are from the logged stdout lines (40 log-spaced points so far), not run.json.
