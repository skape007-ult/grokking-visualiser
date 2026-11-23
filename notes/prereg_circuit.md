# Pre-registration: does the model use its key Fourier frequencies?

Written 2026-11-23, before any of the runs or analyses below were made. The
only code run before this file was frozen was a 300-step timing check
(~19 s per 300 steps on this machine with default threads) and a check that
`--checkpoints` leaves training unchanged (§7). Amendments after review are
in §10 and are dated. Anything decided after seeing results will be marked
**post hoc** in the results file.

## 1. Hypothesis

H: the trained baseline model (`(a + b) mod 97`, weight decay 1.0, 30% of
pairs) computes its answer through the key Fourier frequencies of its
embedding, as in the Fourier-multiplication algorithm of Nanda et al. (2023):
its logits are, to a good approximation, Σ_k α_k cos(2πk(a + b − c)/p) over
a few key frequencies k, and its performance depends on those frequencies
and not on others.

**What this test can and cannot show.** Every measurement here is at the
level of the logits. It can show that the model's output depends on the key
frequencies as functions of (a + b) and on little else. It cannot show *how*
the network produces those terms inside (the trig-identity products in the
MLP neurons). Other algorithms that also produce key-frequency logits (for
example the "pizza" algorithm of Zhong et al. 2023) would pass the same
tests. A pass therefore supports "uses its key frequencies", not "implements
the clock/Fourier-multiplication mechanism" specifically.

## 2. Runs

- **Config:** `baseline` from `train/config.py`, unchanged (p = 97, wd 1.0,
  train_frac 0.3, 25,000 full-batch AdamW steps, CPU, float64 loss,
  deterministic algorithms).
- **Command:** `.venv/bin/python train/train.py --run baseline --seed S --checkpoints --tag circuit__seedS`
- **Seeds:** 0, 1, 2 (the README seeds), then 3 and 4 (added for power, §8).
- **Machine:** this Linux cloud container, 4 CPU cores, Python 3.13,
  torch 2.15.0 (PyPI wheel), default thread count (4). Runs are made one at
  a time, so the thread count is the same for every run.
- **Checkpoints:** the M3 runs saved only embedding snapshots, not weights,
  so the runs must be regenerated. `--checkpoints` (added on this branch)
  saves the full `state_dict` at every recorded metric step (~210 per run,
  dense through the transition).

### 2a. Reproduction check (before any analysis)

The README's grokking steps are 9,868 / 5,785 / 4,216 for seeds 0 / 1 / 2 on
the M3. `notes/repro.md` showed a different Linux CPU gives different steps
(within 5.8%, same seed order), so an exact match is not expected here.

- **Match** = same seed order (seed 0 slowest, seed 2 fastest) **and** each
  seed's grokking step within 10% of the README value.
- If it matches: proceed; every result is labelled as coming from the Linux
  regenerated runs, not the M3 runs on the site.
- If it does not match: still run the analysis (the question is about
  mechanism, not timing), but the mismatch is the first caveat in the
  results file and is flagged in `notes/owner_review.md`.
- Also reported: each seed's key frequencies on these runs vs. the M3 key
  frequencies in `notes/fourier.md`.

## 3. Definitions

**Key frequencies (fixed per seed before any ablation).** The 5 frequencies
with the most energy in the mean-centred number-token rows of `W_E` at the
final step (25,000), computed exactly as in `notes/fourier_analysis.py`
(`TOP = 5`). This is the same "key frequencies" definition as the merged
Fourier note, so the hypothesis is about the frequencies that note found.
*Secondary, reported only:* the top 5 frequencies of the final logits'
(a + b)-spectrum (below), and their overlap with the embedding key set.

**Logit Fourier basis over inputs.** Compute logits L ∈ R^{p²×p} for all
p² = 9,409 inputs (a, b) (train and test together). Over the full grid,
(a + b) mod p takes each value exactly p times, so the vectors

- u = constant, and
- c_k = cos(2πk(a + b)/p), s_k = sin(2πk(a + b)/p), k = 1..48,

each normalised to unit length over the p² inputs, are orthonormal. For a set
S of frequencies, P_S is the orthogonal projection onto span{c_k, s_k : k ∈ S},
applied to each output column of L. The constant u gives each output c its
mean logit over inputs.

**Restricted logits** (keep only S): L_R = P_u L + P_S L.
**Excluded logits** (remove S): L_X = L − P_S L.

Note the component of output c along c_k, s_k is exactly how a term
α cos(2πk(a + b − c)/p) appears, so these are Nanda et al.'s restricted and
excluded logits. **Deviation from Nanda:** they build the restricted logits
from projected MLP neuron activations passed through W_out W_U; this
projects the logits directly. The two differ only by the attention/direct
paths' (a + b)-dependent parts outside S, which the restricted version here
drops as well. The projection is always computed over the full grid, then
losses are evaluated on subsets.

**Metrics** (all cross-entropy in float64, nats; accuracy = argmax):

| name | logits | evaluated on |
|---|---|---|
| restricted loss / acc | L_R with S = key | test set (6,586 pairs) |
| excluded loss / acc | L_X with S = key | train set (2,823 pairs) |
| random-restricted, random-excluded | same, with S = random set | same sets |
| embedding key energy | key-frequency energy fraction of centred W_E | — |

Nanda reports restricted loss on all data and excluded loss on train data.
Restricted loss on the **test** set is used here so it cannot be met by
memorisation; restricted loss on all data and the full-data accuracy are
also reported.

**Random control.** A random set has 5 frequencies drawn uniformly without
replacement from the 43 non-key frequencies. Final-step tests use 100
random sets per seed, drawn with `numpy.random.default_rng(1000 + seed)`.
Over-training curves use the first 5 of those 100 sets.

**Milestones** (from the run's own recorded test accuracy, first recorded
step at or above the threshold, as `first_step` in `train/plot.py`):
t_mem (train acc ≥ 99%), t10, t50, t95 (= grokking step), t99 (test acc).

**Phases:** *before the jump* = t < t10; *during* = t10 ≤ t ≤ t99;
*after* = t > t99.

**Progress and events** for a measure m(t) evaluated at every checkpoint
with t ≥ t_mem: progress p(t) = (m(t) − m(t_mem)) / (m(25,000) − m(t_mem)).
- *onset* = first checkpoint with p ≥ 0.10
- *completion* = first checkpoint with p ≥ 0.90
- *steepest* = the pair of consecutive checkpoints with the largest
  |Δm| / Δln(t); its event time is the geometric mean of the two steps.

Measures use their natural linear scale: loss in nats, energy as a fraction.
(Using log-loss would move onsets earlier; that choice is fixed here.)

## 4. Predictions and thresholds

All final-step tests use the checkpoint at step 25,000.

**Sanity checks (must pass, else the analysis code is wrong):**
- S1. L_X with S = ∅ equals L (max abs diff < 1e-9), and its train/test loss
  equals the training log's final values to 1e-6 relative.
- S2. Keeping u and all 48 frequencies (every function of (a + b)) gives
  full-data accuracy ≥ 99% at step 25,000 (the final model is ~100% on all
  data, so its logits should be close to functions of a + b).
- S3. The basis is orthonormal over the grid (max |FᵀF − I| < 1e-10).

**P1 — key frequencies are necessary (final step).** For every seed:
- excluded train loss ≥ ln 97 ≈ 4.57 (worse than uniform guessing), **and**
- it is larger than the excluded train loss for all 100 random sets
  (rank p ≤ 1/101 per seed), **and**
- for at least 95 of 100 random sets, random-excluded train accuracy is
  ≥ 99%.
Excluded *accuracy* with key frequencies removed is reported but has no
threshold: many tiny coherent terms in other frequencies could still put the
argmax on the right answer even when the loss is large.

**P2 — key frequencies are sufficient (final step).** For every seed:
- restricted test accuracy ≥ 99%, **and**
- restricted test loss is smaller than the random-restricted test loss for
  all 100 random sets, **and**
- for at least 95 of 100 random sets, random-restricted test accuracy is
  ≤ 5% (chance is 1.03%).

**P3 — restricted loss improves before test loss** (Nanda's main progress
measure claim). Restricted test accuracy first reaches 50% at a checkpoint
strictly earlier than t50.

**P4 — the model depends on the key frequencies before it generalises.**
Excluded train accuracy first falls below 50% (at a checkpoint after t_mem)
strictly earlier than t50.

**P5 — phases line up with the Fourier-sparsity timing.** For each of three
measures (restricted test loss, excluded train loss, embedding key energy,
all recomputed on these runs at every checkpoint):
- onset is *before the jump* (< t10),
- steepest is *during* (t10 ≤ steepest ≤ t99),
- completion is *after* (> t99).
That is 3 measures × 3 events = 9 checks per seed.

**P6 — the random control shows no such structure.** Across the 5 random
sets per seed over training: random-restricted test accuracy stays ≤ 5% at
every checkpoint, and random-excluded train accuracy stays ≥ 99% at every
checkpoint from t_mem on.

**Pass rules.** P1, P2 and S1–S3 must hold in **every** seed. P3, P4, P6
pass if they hold in at least 4 of 5 seeds. Each of the 9 P5 checks passes
if it holds in at least 4 of 5 seeds; P5 as a whole "lines up" only if all
9 pass. Every seed's value is reported whatever the rule says.

## 5. What each outcome means

| outcome | reading |
|---|---|
| P1 and P2 pass | At the logit level, the final model's performance is carried by its key frequencies and not by a same-size random set. Supports H at the logit level; says nothing about the internal mechanism (§1). |
| P1 passes, P2 fails | The key frequencies are needed but not enough: other components of the logits matter. H is incomplete. |
| P2 passes, P1 fails | The key frequencies suffice but are redundant with something else (removing them is survivable). Against "uses *through* its key frequencies" as the only route. |
| P1 and P2 fail | Against H. |
| A random-set check fails (random removal also breaks it, or random keeping also works) | The ablation is not specific; the corresponding P is inconclusive rather than passed. |
| S1–S3 fail | Bug; no result is reported until fixed. |
| P3/P4 pass | The key-frequency computation exists and is used on the training set before test accuracy reaches 50%, consistent with Nanda's "circuit formation before cleanup". |
| P3/P4 fail | The key-frequency route appears no earlier than generalisation itself; against the progress-measure story for this model. |
| P5 all pass | The logit-level measures move in the same phases as the embedding sparsity found earlier. |
| P5 partly fails | Report which measure and event differ; that is the finding. No re-definition of phases or progress after the fact. |

## 6. Analysis outputs

`notes/circuit/` will hold: `circuit_analysis.py` (all computation,
standalone, reads `outputs/circuit__seedS/`), `circuit_final.csv` (one row
per seed × condition at step 25,000, including all 100 random sets),
`circuit_over_training.csv` (one row per seed × checkpoint), `circuit_events.csv`
(milestones and P3–P5 events per seed), `circuit_repro.csv` (§2a), and one
figure. The results draft is `notes/RESULTS_circuit_draft.md`. Raw outputs
and checkpoints stay in gitignored `outputs/`.

## 7. Code changes

`train/train.py` gains an opt-in `--checkpoints` flag. It copies the
`state_dict` at recorded steps and writes `checkpoints.pt`. It reads the
model only. Verified before this file was frozen: a 500-step run with and
without the flag gives identical `run.json` metrics and identical
`embeddings.npy` (§10 records the result).

## 8. Power

- **Final-step tests (P1, P2)** are per seed, against 100 random sets each:
  if the key set were no different from a random one, the chance it beats
  all 100 is 1/101 ≈ 0.01 per seed. These have good power with any number of
  seeds.
- **Timing and ordering (P3–P5)** have one yes/no outcome per seed. With 3
  seeds, "holds in all 3" has probability 1/8 = 0.125 under a coin-flip null,
  which is not evidence. With 5 seeds, "holds in at least 4 of 5" has
  probability 6/32 ≈ 0.19 and "all 5" has 1/32 ≈ 0.03. For the 3-way phase
  checks in P5, a uniform-phase null gives ≈ 0.045 for "at least 4 of 5".
- **Decision:** add seeds 3 and 4 (5 in total). At ~26 min per run this is
  ~2.2 h of CPU, which fits. Timing results with 5 seeds remain descriptive:
  they cannot characterise a distribution, and the nulls above are crude
  (phases are not equally likely a priori; "before the jump" covers far
  more log-steps than "during"). The results file will say this.
- Seeds 3 and 4 have no M3 reference values, so they are not part of the
  reproduction check.

## 9. Not tested

- The internal mechanism: neuron activations as products of cos/sin terms,
  attention patterns, W_out/W_U structure.
- Nanda's neuron-level restricted loss (the logit-level version is used).
- Weight-norm "cleanup" measurements.
- Other runs (`no_decay`, `more_data`, `less_data`) and the M3 runs shown on
  the site (no weights were saved there).

## 10. Review and amendments

(Filled in after the independent review, before any run.)
