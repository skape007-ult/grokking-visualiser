# Weight-decay sweep on the M3: when does grokking turn on?

2026-11-08. Branch `exp/wd-sweep-m3`. Raw table: [`wd_sweep.csv`](wd_sweep.csv). Plot: [`wd_sweep.png`](wd_sweep.png). Script: [`wd_sweep.py`](wd_sweep.py).

This replaces the earlier sweep, which mixed cloud-CPU runs with M3 results. Every row below comes from the same Apple M3 MacBook.

## Reproduction check

The three wd = 1.0 runs were retrained rather than reused, and they **match the README exactly**:

| seed | this sweep | README | difference |
|---|---|---|---|
| 0 | 9,868 | 9,868 | 0 |
| 1 | 5,785 | 5,785 | 0 |
| 2 | 4,216 | 4,216 | 0 |

The README lists these sorted as 4,216 / 5,785 / 9,868. By seed they run 9,868 / 5,785 / 4,216. Each run's recorded steps and test-accuracy curve also equal the committed `site/data/runs/baseline.json` at its 4-dp rounding.

## Method

Each sweep run is the `baseline` config from `train/config.py` (1-layer transformer, `(a + b) mod 97`, 30% of pairs for training, full-batch AdamW lr 1e-3, 25,000 steps) with only the weight decay changed:

`.venv/bin/python train/train.py --run baseline --seed S --set weight_decay=W --tag wdW__seedS`

- **Runs:** wd ∈ {0.03, 0.1, 0.3, 0.5, 1.0} × seeds {0, 1, 2}, 15 runs.
- **Device and software:** CPU (the config default, as in the README), Python 3.14.9, torch 2.15.0. No code or config defaults were changed.
- **Execution:** the first run (wd = 1.0, seed 0) ran alone to time it, at 259 s of training. The other 14 ran 4 at a time and took 54 min. The `seconds` column in the CSV therefore reflects how many runs shared the machine, and it is not comparable between runs.
- **wd = 0:** the committed `no_decay` results from this machine, read from `site/data/runs/no_decay.json` and not rerun.
- **Grokking step:** the repo's definition, the first *recorded* step with test accuracy ≥ 95% (`first_step` in `train/plot.py`).
- **"Near the budget":** a run that groks in the last 10% of the step budget (step ≥ 22,500).

## Results

| weight decay | seed | source | grokking step (test acc ≥ 95%) | final train acc | final test acc |
|---|---|---|---|---|---|
| 0 | 0 | committed `no_decay` | did not grok within 25,000 steps | 100.00% | 3.02% |
| 0 | 1 | committed `no_decay` | did not grok within 25,000 steps | 100.00% | 4.05% |
| 0 | 2 | committed `no_decay` | did not grok within 25,000 steps | 100.00% | 4.33% |
| 0.03 | 0 | this sweep | did not grok within 25,000 steps | 100.00% | 3.96% |
| 0.03 | 1 | this sweep | did not grok within 25,000 steps | 100.00% | 5.01% |
| 0.03 | 2 | this sweep | did not grok within 25,000 steps | 100.00% | 6.18% |
| 0.1 | 0 | this sweep | did not grok within 25,000 steps | 100.00% | 5.28% |
| 0.1 | 1 | this sweep | did not grok within 25,000 steps | 100.00% | 9.00% |
| 0.1 | 2 | this sweep | did not grok within 25,000 steps | 100.00% | 10.20% |
| 0.3 | 0 | this sweep | did not grok within 25,000 steps | 100.00% | 17.93% |
| 0.3 | 1 | this sweep | 20,112 | 100.00% | 99.95% |
| 0.3 | 2 | this sweep | 18,218 | 100.00% | 100.00% |
| 0.5 | 0 | this sweep | 20,514 | 100.00% | 99.92% |
| 0.5 | 1 | this sweep | 11,790 | 100.00% | 100.00% |
| 0.5 | 2 | this sweep | 9,485 | 100.00% | 100.00% |
| 1 | 0 | this sweep | 9,868 | 100.00% | 100.00% |
| 1 | 1 | this sweep | 5,785 | 100.00% | 100.00% |
| 1 | 2 | this sweep | 4,216 | 100.00% | 100.00% |

No run is marked "near the budget". The latest grok is step 20,514, which is 82% of the way through the 25,000 steps.

![grokking step vs weight decay](wd_sweep.png)

## Reading

Within 25,000 steps, no seed groks at wd ≤ 0.1, two of three grok at wd = 0.3, and all three grok at wd = 0.5 and 1.0, so grokking turns on between wd = 0.1 and wd = 0.5 for this budget. Where every seed groks, a larger weight decay shortens the delay: the median grokking step is 11,790 at wd = 0.5 and 5,785 at wd = 1.0. Seed 0 at wd = 0.3 did not grok within 25,000 steps, but its test accuracy was still climbing (11% near step 20,000, 18% at the end), so the budget limits what this sweep can say close to the switch-on point. Weight decay already has a visible effect below that point, with final test accuracy rising from 3–4% at wd = 0 to 4–6% at wd = 0.03 and 5–10% at wd = 0.1. Only wd = 0.5 and 1.0 have a grokking step for every seed, so this sweep does not establish how the delay scales with weight decay.

## Limitations

- **Three seeds per setting.** At wd = 0.5 the grokking step varies 2.2× across seeds, and at wd = 1.0 it varies 2.3×. That locates the switch-on point only roughly and does not characterise a timing distribution.
- **Seeds are not independent across weight decays.** A seed fixes both the train/test split and the initialisation, so the same seed is reused at every wd. Seed 0 is the slowest at wd = 0.3, 0.5 and 1.0, and seed 2 is the fastest. Treat differences between wd values as paired comparisons on three seeds.
- **"Did not grok" means only "not within 25,000 steps".** All 7 non-grokking runs in this sweep had higher test accuracy at step 25,000 than near step 20,000, so a longer budget might show late grokking.
- **One task, one configuration:** addition mod 97, 30% training data, one architecture, one learning rate. The switch-on point probably depends on all of these.
- **One machine and software stack.** The numbers are for this M3 with torch 2.15.0 on CPU. The README notes that a ~1e-8 numerical change once shifted a grokking step by 5%, so these exact steps should not be expected on other hardware.
- **The wd = 0 rows were not retrained.** They come from committed data rounded to 4 dp. The reproduction check above covers wd = 1.0 only.
- **Recorded steps only.** The grokking step is quantised to the recording schedule, about 2–5% of the step value.

## Appendix: Earlier cloud run

The previous version of this note (branch `exp/wd-sweep`, 2026-11-05) ran 6 of these settings on a 4-core Linux CPU with 1–2 threads per run. wd = 0.03 was not completed there. These numbers are kept for reference only and are not comparable to the M3 rows; the M3 columns are repeated here to show the size of the difference.

| weight decay | seed | cloud grokking step | M3 grokking step | cloud final test acc | M3 final test acc |
|---|---|---|---|---|---|
| 0.1 | 0 | did not grok within 25,000 steps | did not grok within 25,000 steps | 5.50% | 5.28% |
| 0.1 | 1 | did not grok within 25,000 steps | did not grok within 25,000 steps | 8.11% | 9.00% |
| 0.3 | 0 | did not grok within 25,000 steps | did not grok within 25,000 steps | 12.80% | 17.93% |
| 0.3 | 1 | 19,718 | 20,112 | 99.95% | 99.95% |
| 0.5 | 0 | 20,514 | 20,514 | 100.00% | 99.92% |
| 0.5 | 1 | 11,560 | 11,790 | 100.00% | 100.00% |
