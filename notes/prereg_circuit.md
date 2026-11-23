# Pre-registration: does the model use its key Fourier frequencies?

- **Version 1** was written 2026-11-23 before any run (commit e4f564e).
- **Version 2** (this file) answers the independent review in §10. It was
  frozen before any analysis of the circuit runs; training had started but
  no output had been read.
- Code run before v1 was frozen: a 300-step timing check (~19 s per 300 steps
  at default threads, ~54 s at 1 thread) and the 500-step `--checkpoints`
  identity check (§7).
- Code run between v1 and v2: an 800-step `more_data` smoke test of the
  analysis script (not a baseline run, discarded). Its output was seen. No
  threshold below was set from it, and every change in v2 traces to a review
  finding in §10.
- Anything decided after seeing circuit results is marked **post hoc** in the
  results file.

## 1. Hypothesis

H: the trained baseline model (`(a + b) mod 97`, weight decay 1.0, 30% of
pairs) computes its answer through the key Fourier frequencies of its
embedding, as in the Fourier-multiplication algorithm of Nanda et al. (2023).
Its logits are, to a good approximation, Σ_k α_k cos(2πk(a + b − c)/p) over a
few key frequencies k, and its performance depends on those frequencies and
not on others.

**What this test can and cannot show.** Every measurement here is at the
level of the logits. It can show that the model's output depends on the key
frequencies as functions of (a + b), and on little else. It cannot show *how*
the network produces those terms inside (the trig-identity products in the
MLP neurons). Other algorithms that also produce key-frequency logits, for
example the "pizza" algorithm of Zhong et al. (2023), would pass the same
tests. A pass therefore supports "uses its key frequencies". It does not
support "implements the clock / Fourier-multiplication mechanism"
specifically.

## 2. Runs

- **Config:** `baseline` from `train/config.py`, unchanged: p = 97, wd 1.0,
  train_frac 0.3, 25,000 full-batch AdamW steps, CPU, float64 loss,
  deterministic algorithms.
- **Command:** `.venv/bin/python train/train.py --run baseline --seed S --checkpoints --tag circuit__seedS`
- **Seeds:** 0, 1 and 2 (the README seeds), then 3 and 4, added for power
  (§8).
- **Machine:** this Linux cloud container, 4 CPU cores, Python 3.13,
  torch `2.15.0+cu132` (PyPI wheel, run on CPU), default thread count (4).
  Runs are made one at a time, so every run uses the same thread count.
- **Checkpoints:** the M3 runs saved only embedding snapshots, not weights, so
  the runs must be regenerated. `--checkpoints` (added on this branch) saves
  the full `state_dict` at every recorded metric step: about 210 per run,
  dense through the transition.

### 2a. Reproduction check (before any analysis)

On the M3, the README's grokking steps are 9,868 / 5,785 / 4,216 for seeds
0 / 1 / 2. `notes/repro.md` showed that a different Linux CPU gives different
steps: within 5.8%, in the same seed order. So an exact match is not expected
here. The 10% tolerance below was set after seeing that 5.8% figure.

- **Match** means the same seed order (seed 0 slowest, seed 2 fastest)
  **and** each seed's grokking step within 10% of its README value.
- **If it matches:** proceed. Every result is labelled as coming from the
  regenerated Linux runs, not from the M3 runs shown on the site.
- **If it does not match:** still run the analysis, since the question is
  about mechanism, not timing. The mismatch becomes the first caveat in the
  results file and is flagged in `notes/owner_review.md`. A mismatch changes
  no other rule.
- **Also reported:** each seed's key frequencies on these runs, against the
  M3 key frequencies in `notes/fourier.md`.

## 3. Definitions

**Key frequencies (fixed per seed before any ablation).**
- **Definition:** the 5 frequencies with the most energy in the mean-centred
  number-token rows of `W_E` at the final step (25,000), computed with the
  same rule as `notes/fourier_analysis.py` (`TOP = 5`).
- **These runs:** each regenerated run uses its *own* key set. Because seed
  1's set already changed across hardware, these may not be the M3 sets in
  `notes/fourier.md`; the overlap is reported.
- **Logit-defined set (secondary):** the top 5 frequencies of the final
  logits' (a + b)-spectrum (below). P1 and P2 are also run with this set, as
  a secondary result.

**Inputs and split.** All p² = 9,409 inputs (a, b), indexed as row a·p + b
(`indexing="ij"`). The train and test masks come from the seed's permutation.
The script asserts that the grid rows selected by these masks reproduce
`make_dataset`'s `train_x` and `test_x` exactly (same rows, same order).

**Logit Fourier basis over inputs.**
- **Basis vectors:** over the full grid, (a + b) mod p takes each value
  exactly p times, so these vectors are orthonormal:
  - u, the constant vector;
  - c_k = cos(2πk(a + b)/p) and s_k = sin(2πk(a + b)/p), for k = 1..48;
  - each normalised to unit length over the p² inputs.
- **Logits:** L ∈ R^{p²×p}, one column per output c.
- **Why per output:** a term α cos(2πk(a + b − c)/p) in output c's column
  lies exactly in span{c_k, s_k}.

**Restricted logits (keep S): two versions.**
- **Primary, test-fit:** L_R^test. On the **test rows only**, least-squares
  fit each output column onto [u, c_k, s_k for k ∈ S] restricted to those rows
  (via QR). Evaluate the fitted values on the same test rows. Train rows never
  enter, so memorised train logits cannot leak in.
- **Secondary, full-grid:** L_R^grid = P_u L + P_S L, where P_S is the
  orthogonal projection over the full grid. This is reported but **labelled
  as leaking**. A memorised spike at the train pairs with a + b = c has energy
  at every frequency, so its projection peaks at the right answer on test rows
  too (review finding 1).

**Excluded logits (remove S):** L_X = L − P_S L, using the full-grid
projection (Nanda's construction), and evaluated on the **train** rows.

**Scale-free versions.** This model's logit scale is very large: test loss
is ~20 nats on the plateau, and train loss falls below 1e-6 at the end. Raw
cross-entropy therefore mostly tracks scale, so the primary loss-type metrics
are scale-free:
- **Calibrated CE:** CE*(L) = min over τ > 0 of CE(L/τ).
  - τ is found by bounded scalar minimisation over log τ ∈ [−12, 12].
  - It lies in [0, ln 97], and equals ln 97 when L carries no information.
- **Key margin share:** φ_S = mean_rows[(P_S L)_y − mean_c (P_S L)] /
  mean_rows[L_y − mean_c L], evaluated on train rows with the full-grid
  projection. It is the share of the model's average correct-answer margin on
  the training set that is carried by S.

Raw CE is always reported alongside.

**Metrics** (CE in float64, nats; accuracy is argmax):

| name | logits | rows |
|---|---|---|
| restricted acc, CE* (primary) | L_R^test, S = key | test |
| restricted acc, CE (secondary, leaking) | L_R^grid, S = key | test and all |
| excluded acc, CE* | L_X, S = key | train |
| key margin share φ | P_S L | train |
| random-set versions of the above | S = random set | same rows |
| embedding key energy | key-frequency energy fraction of centred W_E | none (computed on W_E) |
| negative control | the model's own logits (CE*) | test |

**Controls** (final step unless stated otherwise):

- **Random sets.**
  - 5 frequencies drawn uniformly without replacement from the 43 non-key
    frequencies; 100 sets per seed, from
    `numpy.random.default_rng(1000 + seed)`.
  - The first 5 of those sets are also tracked over training.
  - Because the key set can never be drawn, a rank of "beats all 100" is
    **descriptive**, not a calibrated p-value.
- **Harder sets** (secondary, final step only):
  - (i) the frequencies ranked 6–10 by the final logits' (a + b)-spectrum,
    which catches harmonics made by the ReLU and invisible in W_E;
  - (ii) each other seed's key set;
  - (iii) dose-response: the key set minus one frequency (5 sets), and the
    key set's top 4 alone.

**Milestones.** From the run's own recorded test accuracy: the first
recorded step at or above the threshold, as in `first_step` in
`train/plot.py`.
- **t_mem:** train acc ≥ 99%.
- **t10, t50, t95 (the grokking step) and t99:** test acc ≥ 10%, 50%, 95%
  and 99%.

**Phases:**
- *before the jump:* t < t10;
- *during:* t10 ≤ t ≤ t99;
- *after:* t > t99.

**Events for P5**, computed for a measure m(t) over checkpoints with
t ≥ t_mem:

- **Expected direction.** Each measure has one: calibrated restricted CE
  falls, calibrated excluded CE rises, embedding key energy rises.
- **Progress:** p(t) = (m(t) − m(t_mem)) / (m(25,000) − m(t_mem)).
  - If |m(25,000) − m(t_mem)| < 0.1 nats (0.05 for energy), or the change
    runs opposite to the expected direction, the events are **undefined**.
    The check then fails and the result is reported.
- **Onset:** the first checkpoint with p ≥ 0.10.
- **Completion:** the first checkpoint after which p stays ≥ 0.90 at every
  later checkpoint.
- **Steepest:**
  - Resample m onto a fixed grid in ln t, spacing 0.05, from t_mem to 25,000,
    by linear interpolation in ln t.
  - Find the largest *signed* change per grid step in the expected direction.
  - The event time is the geometric mean of that grid interval's two ends.
- **Sensitivity:** the same events, with the endpoint at step 20,000
  instead of 25,000, are reported (no pass rule).
- **Effect sizes:** reported per seed as ln t_event − ln t10, ln t_event −
  ln t99 and ln t_event − ln t50.

## 4. Predictions and thresholds

All final-step tests use the checkpoint at step 25,000.

**Implementation checks.** These must pass; if one fails, the analysis code
is wrong and nothing is reported until it is fixed.

- **C1:** L_X with S = ∅ equals L exactly.
- **C1b:** the analysis computes train and test loss on `make_dataset`'s own
  tensors, in the same way as `evaluate` in train.py. These must equal
  run.json's final values (relative difference < 1e-6).
- **C1c:** grid logits at the train rows equal logits computed on `train_x`
  (max abs difference < 1e-4).
- **C2:** synthetic test. Logits Σ_k α_k cos(2πk(a + b − c)/p), plus a
  non-(a + b) term f(a) + g(b, c), plus noise. Then:
  - the full-grid restricted logits recover the constant plus the S part to
    1e-9;
  - the excluded logits remove exactly the S part.
- **C3:** max |FᵀF − I| < 1e-10.
- **C4:** the grid rows and masks reproduce `make_dataset`'s `train_x` and
  `test_x` exactly.

**P0. The final logits are mostly a function of (a + b) (reported, with a
threshold).** At step 25,000, keeping u and all 48 frequencies with the
test-fit restricted logits gives test accuracy ≥ 99%.

**P1. The key frequencies are necessary (final step).** For every seed:
- with the key set excluded, train accuracy is ≤ 10%, **and**
- with the key set excluded, calibrated train CE* is ≥ 0.9 · ln 97 (4.12),
  **and**
- for at least 95 of the 100 random sets, train accuracy with that set
  excluded is ≥ 99%.

Reported without a threshold:
- the raw excluded CE;
- whether the key-excluded CE* exceeds every random set's (a descriptive
  rank);
- all the harder controls.

**P2. The key frequencies are sufficient (final step).** For every seed:
- test-fit restricted test accuracy with the key set is ≥ 99%, **and**
- for at least 95 of the 100 random sets, test-fit restricted test accuracy
  is ≤ 5% (chance is 1.03%).

Reported without a threshold:
- calibrated CE* against the random sets (a descriptive rank);
- the full-grid (leaking) version;
- the harder controls.

**P3. The restricted measure improves before test accuracy does.** Both of
these happen at checkpoints strictly earlier than t50:
- test-fit restricted test accuracy (key set) first reaches 50%, **and**
- key-set restricted test accuracy minus the maximum over the 5 tracked
  random sets first reaches 0.30.

**P4. The model's training-set answers are carried by the key frequencies
before it generalises** (circuit formation). The key margin share φ_key on
train rows first reaches 0.5 at a checkpoint strictly earlier than t50.

**P4-report (no prediction).** The first checkpoint after t_mem at which
train accuracy with the key set excluded falls below 50%. This measures when
memorisation is *removed* (cleanup), not when the circuit forms. It is
reported alongside t50 and t95.

**P5. The phases line up with the Fourier-sparsity timing.** Measures:
1. calibrated restricted test CE* (test-fit, key set);
2. calibrated excluded train CE* (key set);
3. embedding key energy.

For each measure:
- onset is *before the jump* (before t10);
- steepest is *during* the jump (t10 ≤ steepest ≤ t99);
- completion is *after* the jump (after t99).

This gives 9 checks per seed.

Status of these checks:
- **Measure 3 is a replication, not a test.** Its thresholds were set with
  the M3 values in `notes/fourier.md` visible: its onset progress just before
  t10 was 0.106 / 0.122 / 0.107, and seed 2's steepest rise ended near t99.
  Its completion check is expected **by construction**, because the key set
  is chosen at step 25,000. The margin to every threshold is reported.
- **Negative controls:** the same event code runs on the model's own test
  CE* and on each of the 5 random-set curves. If the model's own test CE*
  also shows "onset before t10", the onset check cannot discriminate, and the
  results file says so.

**P6. The random control shows no such structure** over training, for each
of the 5 tracked random sets and at ≥ 95% of the checkpoints from t_mem on:
- random-set test-fit restricted test accuracy is ≤ 5%, **and**
- train accuracy with the random set excluded is within 1 percentage point
  of the model's own train accuracy.

**Pass rules.**
- **C1–C4, P1 and P2:** must hold in **every** seed.
- **P0, P3, P4 and P6:** pass if they hold in at least 4 of 5 seeds.
- **P5:** each of the 9 checks passes if it holds in at least 4 of 5 seeds.
  P5 "lines up" only if all 6 checks on measures 1 and 2 pass. Measure 3 is
  reported as a replication.
- Every seed's value is reported, whatever the rule says.

**Secondary analyses (no pass rules).**
- P1 and P2 repeated with the logit-defined key set.
- The non-MLP-path part of the logits inside S at the final step, reported
  as ‖P_S(L − L_mlp)‖_F / ‖P_S L‖_F, where L_mlp is the logit contribution of
  the MLP output (relu(·) W_out + b_out) W_U.

## 5. What each outcome means

| outcome | reading |
|---|---|
| P1 and P2 pass | At the logit level, the final model's performance is carried by its key frequencies, not by a same-size set of non-key frequencies. This supports H at the logit level and says nothing about the internal mechanism (§1). |
| P1 passes, P2 fails | The key frequencies are needed but not enough: other components of the logits matter. H is incomplete. |
| P2 passes, P1 fails | The key frequencies suffice but are redundant: something else also carries the answer, so removing them is survivable. This is against "through its key frequencies" being the only route. |
| P1 and P2 fail | Against H. |
| A random-set part fails (removing random sets also breaks it, or keeping random sets also works) | The ablation is not specific, so that P is inconclusive rather than passed. |
| Embedding and logit key sets overlap in ≤ 3 of 5 | The embedding set is the wrong set at the logit level, for example because a harmonic carries logit energy. P1/P2 with the embedding set are then read alongside the logit-set results. If only the logit set passes, the claim is "uses the logits' frequencies", not "uses the embedding's key frequencies". |
| C1–C4 fail | A bug. No result is reported until it is fixed. |
| P0 fails | The final logits contain substantial non-(a + b) structure, so the restricted-logit framing loses information. Report that and read P2 with caution. |
| P3 passes | Using only the key frequencies, the model generalises before its full logits do: the memorisation components mask a circuit that already works. |
| P3 fails | The key-frequency route works no earlier than the model's own test accuracy. This is against the progress-measure story for this model. |
| P4 passes | Most of the training-set margin is already carried by the key frequencies before test accuracy reaches 50%: circuit formation before generalisation. |
| P4 fails | Train answers are still carried mostly by other components at t50. |
| P5, measures 1–2, all pass | The logit-level measures move in the same phases as the embedding sparsity found earlier. |
| P5 partly fails | Report which measure and event differ; that is the finding. Phases and progress are not redefined after the fact. |
| A negative control has the same P5 pattern | That event check does not discriminate. Say so, and do not count it as support. |

## 6. Analysis outputs

All under `notes/circuit/`:

- `circuit_analysis.py`: all computation, standalone; reads
  `outputs/circuit__seedS/`.
- `circuit_repro.csv`: the §2a check.
- `circuit_keyfreqs.csv`: the key sets.
- `circuit_final.csv`: every condition at step 25,000, including all 100
  random sets and the harder sets.
- `circuit_over_training.csv`: one row per seed × checkpoint.
- `circuit_events.csv`: milestones, events, effect sizes and sensitivity.
- `circuit_predictions.csv`: every check per seed, with pass/fail and the
  numbers.
- One figure.

Elsewhere:
- The results draft is `notes/RESULTS_circuit_draft.md`.
- Raw outputs and checkpoints stay in gitignored `outputs/`.

## 7. Code changes

`train/train.py` gains an opt-in `--checkpoints` flag.
- It copies the `state_dict` at every recorded step and writes
  `checkpoints.pt`. It reads the model only.
- **Verified before v1 was frozen**, on 500-step runs with and without the
  flag:
  - all 5 metric arrays (200 entries each) are identical;
  - `embedding_steps` and `embeddings.npy` are identical;
  - there are 200 checkpoints, at exactly the recorded steps;
  - reloading the step-500 checkpoint and recomputing the train loss gives
    exactly the logged value, 0.0006182105367938108.

## 8. Power

- **Final-step tests (P1, P2).** These are per seed, and their thresholds are
  absolute (accuracies, CE*). The comparison with the 100 random sets is
  descriptive, because the key set cannot be drawn (no exchangeability). Five
  seeds are enough for these.
- **Timing and ordering (P3–P5).** Each gives one yes/no outcome per seed.
  - **Under a coin-flip null:** 3 of 3 seeds has probability 0.125; at least
    4 of 5 has 6/32 ≈ 0.19; all 5 has 1/32 ≈ 0.03.
  - **For the 3-way phase checks**, a uniform-phase null gives ≈ 0.045 for
    at least 4 of 5.
  - **These nulls are crude.** The phases are not equally likely a priori:
    "before" spans far more log-steps than "during". So pass/fail counts
    carry little inferential weight.
  - **What will be reported instead:** per-seed effect sizes in ln-steps,
    their range across the 5 seeds, and the negative controls. Those are the
    substance.
- **Decision:** add seeds 3 and 4, for 5 seeds in total. At ~26 min per run,
  that is ~2.2 h of CPU.
  - Timing results with 5 seeds stay descriptive. They cannot characterise a
    distribution.
  - Seeds 3 and 4 have no M3 reference values, so they are not in the
    reproduction check.

## 9. Not tested

- The internal mechanism: neuron activations as products of cos/sin terms,
  attention patterns, and W_out / W_U structure.
- Nanda's neuron-level restricted loss. The logit-level version is used here.
  The non-MLP part inside S is reported, to show how much the two versions
  could differ.
- Weight-norm "cleanup" measurements.
- The other runs (`no_decay`, `more_data`, `less_data`), and the M3 runs shown
  on the site, which saved no weights.
- Clock vs pizza (§1).

## 10. Review and amendments

An independent reviewer (a fresh agent given the v1 file and the code, not my
reasoning) reviewed v1 on 2026-11-23 before any baseline run. Verdict: "ready
to run after the must-fixes (1–3)"; training config needs no change. All 14
findings and the responses:

| # | severity | finding (short) | response |
|---|---|---|---|
| 1 | must-fix | Full-grid projection leaks memorised train logits into the test-set restricted metric: a pure memoriser gets 0.35–0.90 restricted test acc with a random set. | **Adopted.** The primary restricted metric is now fitted by least squares on test rows only. The full-grid version is kept as secondary, labelled as leaking. P3 adds a key-minus-random margin of ≥ 0.30. |
| 2 | must-fix | "Excluded loss ≥ ln 97" is a knife edge decided by logit scale. Dropping the accuracy criterion was backwards: high excluded accuracy *is* evidence against necessity. | **Adopted.** P1 now needs excluded train accuracy ≤ 10% and calibrated CE* ≥ 0.9 ln 97. Raw CE is reported only. |
| 3 | must-fix | P5 linear-loss events are dominated by scale changes, and some events are fixed by construction. | **Adopted.** P5 uses calibrated CE* for the restricted and excluded measures. The embedding-energy completion is marked as by construction. The negative controls (own test CE*, random sets) are added, with the rule for reading them fixed in advance. |
| 4 | should-fix | "Steepest" is biased towards the dense-recording window and ignores sign. | **Adopted.** The measure is resampled onto a fixed ln-t grid (spacing 0.05), and steepest is the signed change in the expected direction. |
| 5 | should-fix | Progress is unstable for curves that do not move steadily one way. | **Adopted.** Completion is now "stays ≥ 0.9". An event is undefined (the check fails, and the result is reported) when the total change is < 0.1 nats (0.05 for energy) or runs the wrong way. The 20,000-step endpoint is reported as a sensitivity row. |
| 6 | should-fix | The embedding-energy P5 checks are not blind (M3 values were known) and sit on a knife edge. | **Adopted.** They are labelled as replication, not test, and the margins are reported. P5 "lines up" depends only on measures 1–2. |
| 7 | should-fix | P4 (excluded acc < 50%) measures cleanup, not circuit formation; §5's reading was wrong. | **Adopted.** P4 is now the key margin share φ ≥ 0.5 before t50. Excluded-accuracy timing moves to P4-report, with no prediction. §5 is reworded. |
| 8 | should-fix | The random control is fair in size but uninformative, and 1/101 assumes exchangeability. | **Adopted.** Added logit-ranked frequencies 6–10, other seeds' key sets, and leave-one-out / top-4 dose-response as secondary controls. The rank is now called descriptive. |
| 9 | should-fix | There is no rule for when the embedding and logit key sets disagree. | **Adopted.** Added an outcome row for overlap ≤ 3. P1/P2 are repeated with the logit-defined set as a secondary result. |
| 10 | should-fix | The deviation-from-Nanda sentence was wrong: the versions differ by the non-MLP components *inside* S. | **Adopted.** Corrected (§3, §9). The non-MLP share of P_S L is reported at the final step. |
| 11 | should-fix | S1's tolerance is fragile, S2 is an empirical claim rather than a code check, and the grid indexing should be asserted. | **Adopted.** C1b recomputes on `make_dataset` tensors. C1c compares grid and batch logits with an absolute tolerance of 1e-4. C2 is now a synthetic unit test. Old S2 is now prediction P0. C4 asserts the indexing. |
| 12 | minor | "Every checkpoint" criteria are brittle at t_mem. | **Adopted.** P6 now holds at ≥ 95% of checkpoints, with excluded accuracy within 1 percentage point of the model's own train accuracy. |
| 13 | minor | The power arithmetic is right but beside the point. Report effect sizes; note that the repro tolerance was set after seeing 5.8%. | **Adopted.** Effect sizes are reported in ln-steps (§3, §8), and the note is added to §2a. |
| 14 | minor | Housekeeping: §10 was empty, the 300- vs 500-step checks were not reconciled, the torch string was inexact, and "the frequencies that note found" was not quite true. | **Fixed** in the header, §2, §3 and §7. |
