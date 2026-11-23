# Does the model use its key Fourier frequencies? (draft, for owner wording)

2026-11-23. Branch `claude/amazing-hamilton-7hqszu-circuit`.

- **Pre-registration:** [`prereg_circuit.md`](prereg_circuit.md). Version 1
  was reviewed independently, and version 2 was frozen before any analysis.
- **Code and data:** [`circuit/`](circuit/).
- **Status:** not on the site.

## Audit outcome

An independent agent was given the code, the pre-registration and the result
files, not these conclusions. It recomputed every pass-deciding number with
its own implementation, using different methods: an FFT-based projection, a
pinv least-squares fit and a grid-search calibrated CE. It covered all 5
seeds, all 100 random sets and every checkpoint.

- **Verdict: PASS.** Every number matches the CSVs to their printed
  precision, and the code matches the pre-registration on every pass-deciding
  definition.
- **Caveats it raised** are carried into the sections below. One concerns
  the secondary analyses: the "logit rank 6–10" control set is mislabelled
  (it is the top 5 logit frequencies outside the key set), and the random
  sets reused for the logit-key repeat can contain logit-key frequencies.
  Those secondary analyses are not used for any claim here.
- **A second, small audit** checked the thread-count comparison in the
  reproducibility section.

Only audited numbers appear in this file.

## Setup in one paragraph

The `baseline` config is unchanged (wd 1.0, 30% of pairs, 25,000 steps).
Seeds 0–4 were retrained on a 4-core Linux CPU with weights saved at every
recorded step. Each seed's **key frequencies** are the 5 with the most energy
in its final embedding, as in `fourier.md`. Write the logits as functions of
the input pair (a, b):

- **Excluded logits** remove each key frequency's cos/sin(2πk(a + b)/97)
  component.
- **Restricted logits** keep only those components and a constant, fitted on
  test pairs only.

Accuracy and a temperature-calibrated cross-entropy (CE*, between 0 and
ln 97 ≈ 4.57) are compared against 100 random sets of 5 non-key frequencies
per seed. All measurements are at the logit level.

## Claims (at most three)

**1. At the end of training, the model's answers depend on its 5 key
frequencies, and not on same-size sets of other frequencies (P1, passed in
5/5 seeds).**

- **Removing the key frequencies:** train accuracy falls to 0.8–5.7% across
  seeds, with chance at 1.03%. CE* is 4.54–4.57, against a threshold of
  0.9·ln 97 = 4.12; no information is ln 97 ≈ 4.57.
- **Removing random sets:** for every one of the 100 random 5-frequency sets,
  in every seed, train accuracy stayed ≥ 99%.

| seed | key frequencies | train acc, key excluded | CE*, key excluded | random sets with train acc ≥ 99% |
|---|---|---|---|---|
| 0 | 3, 10, 12, 20, 32 | 2.4% | 4.563 | 100/100 |
| 1 | 14, 21, 31, 33, 34 | 2.5% | 4.566 | 100/100 |
| 2 | 1, 9, 15, 34, 41 | 5.7% | 4.556 | 100/100 |
| 3 | 1, 31, 34, 35, 48 | 4.9% | 4.538 | 100/100 |
| 4 | 3, 4, 29, 41, 47 | 0.8% | 4.559 | 100/100 |

**2. Using only the key frequencies, the model already generalises before
its full output does (P3 and P4, passed in 5/5 seeds).** In every seed:
- the restricted logits reached 50% test accuracy before the model itself
  did;
- the restricted accuracy led the best of 5 tracked random sets by ≥ 0.30
  even earlier;
- the key frequencies already carried half of the model's average
  training-set margin (φ ≥ 0.5) before test accuracy reached 50%.

| seed | model test acc ≥ 50% (t50) | restricted test acc ≥ 50% | key − random ≥ 0.30 | φ ≥ 0.5 | lead of restricted over t50, in ln-steps |
|---|---|---|---|---|---|
| 0 | 8,939 | 7,050 | 5,785 | 8,424 | 0.24 |
| 1 | 4,938 | 2,521 | 1,631 | 3,671 | 0.67 |
| 2 | 3,528 | 1,697 | 938 | 2,521 | 0.73 |
| 3 | 6,138 | 3,325 | 2,068 | 5,037 | 0.61 |
| 4 | 22,646 | 14,949 | 11,790 | 20,924 | 0.42 |

This is Nanda et al.'s "restricted loss falls before test loss" pattern,
shown here with a restricted fit that cannot use memorised training pairs.
The φ lead is small in seeds 0 and 4 (0.06 and 0.08 ln-steps).

## Failed predictions

- **P2, sufficiency: FAILED by its pre-registered rule.**
  - What passed: keeping only the key frequencies gave **100% test accuracy
    in all 5 seeds**.
  - What failed: the specificity part required ≥ 95 of 100 random sets to
    stay at ≤ 5% test accuracy. That held in seeds 0–2 (99, 100, 96 sets) but
    **not in seeds 3 and 4 (88 and 88)**. The worst random set reached 8.4%.
  - The rule says a failed random-set part makes P2 inconclusive, not passed.
  - The auditor noted that in seed 3, every failing random set contains
    frequency 4. Frequency 4 is in that seed's top 5 *logit* frequencies but
    not in its embedding key set. This is a lead, not a result.
- **P5, phases line up with the embedding-sparsity timing: FAILED** (3 of the
  6 checks on the logit measures failed).
  - **Restricted CE\* completion:** predicted after test accuracy reaches
    99%; it came during the jump in **0/5** seeds. The auditor's point:
    restricted CE* reaches ~0 before t99 in every seed, so this check
    measured saturation, not phase.
  - **Excluded CE\* onset:** predicted before test accuracy reaches 10%; it
    held in **2/5** (seeds 2 and 4). In the other three it came early in the
    jump.
  - **Excluded CE\* completion:** predicted after t99; it held in **1/5**
    (seed 0 only).
  - Passed: restricted onset (5/5), restricted steepest (4/5), excluded
    steepest (5/5).
  - **Reading:** the logit-level measures finish inside the test-accuracy
    jump. The embedding sparsity keeps rising well after it, as `fourier.md`
    found. So these measures do not share the embedding measure's "finishes
    after" phase.
- **Embedding-energy replication (not a test):** steepest during the jump in
  only 3/5 seeds (seeds 2 and 3 were after). Onset was before the jump in
  4/5, and completion was after in 5/5, which is by construction.

## Null and weak results stated plainly

- **The random-set curves gave no usable negative control for the timing
  events.** Their CE* barely moves (changes below 0.04 nats), so their events
  are undefined by design.
- **The "onset before the jump" check is weak.** It needs only a 10% move
  over a window that starts at step ~130. The model's own test CE* has its
  onset during the jump in seeds 0–3, but before it in seed 4. By the
  pre-registered rule, that check does not discriminate in seed 4.
- **Seed 4 grokked near the budget** (95% at step 23,098 of 25,000). Its key
  frequencies hold only 0.84 of its embedding energy at the end, against
  0.975–0.994 in the other seeds, and are still rising. Its key set and its
  completion checks come from an unconverged model.

## What this does and does not show

It shows, at the logit level and in 5 seeds, that the trained model's
answers run through its key frequencies, and that this route works on unseen
pairs before the model's full output does.

It does **not** show the internal mechanism, that is, the neurons computing
cos/sin products. It does not distinguish the clock algorithm from the
"pizza" algorithm, both of which produce key-frequency logits. It also does
not use Nanda's neuron-level restricted loss: the analysis projects the
logits directly. The size of the non-MLP part of the key-frequency logit
component is in `circuit/circuit_predictions.csv`, but it was not audited, so
it is not quoted here.

## Reproducibility

The runs here are on a Linux CPU, not the M3 runs shown on the site.
Pre-registered match rule: same seed order and within 10%. **It matched.**

| seed | README (M3) | earlier cloud run (`repro.md`) | this run (Linux, 4 threads) | this vs README |
|---|---|---|---|---|
| 0 | 9,868 | 10,065 | 10,065 | +2.0% |
| 1 | 5,785 | 5,452 | 5,901 | +2.0% |
| 2 | 4,216 | 3,973 | 4,052 | −3.9% |
| 3 | — | — | 6,913 | — |
| 4 | — | — | 23,098 | — |

**Thread count (exploratory, not pre-registered; independently checked).**
- **The rerun:** seed 1 was rerun with 1 thread instead of 4, with an
  identical config. It differed from the first optimiser step (train loss
  differs by 9e-10 at step 1) and grokked at **6,138 instead of 5,901**.
- **What this shows:** thread count alone changes results on this machine,
  and that tiny floating-point difference moves the grokking step by about
  4%, two points on the recording grid.
- **What it does not show:**
  - whether the earlier cloud run's 5,452 came from its thread count. That
    run's thread count was not recorded, so the explanation for 5,452
    remains **untested**;
  - that thread count controls grokking time in any particular direction.

Neither run records its thread count in `run.json`. The two runs were also
launched differently: the 4-thread run used `--checkpoints`, which copies
weights only.
