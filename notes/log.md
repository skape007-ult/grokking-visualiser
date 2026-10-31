# Work log

Dated log of every training attempt: config (or diff from the previous one),
key numbers, and interpretation. Milestones are read from the ~200 log-spaced
recorded steps, so "step N" means "first *recorded* step at or after which the
threshold held" (recorded steps are ~3–4% apart late in training, e.g.
…, 9 960, 10 400, …).

Milestone definitions used throughout:
- **train≈100%**: first recorded step with train acc ≥ 99%
- **test 50% / 95%**: first recorded step with test acc ≥ 50% / ≥ 95%

---

## 2026-10-30 — Setup

- Environment: Apple M3 (8-core, macOS), Python 3.14, torch 2.15.0, numpy 2.5.4,
  matplotlib 3.11.2, all in a local `.venv`.
- Model follows Nanda et al. 2023: 1 layer, d_model 128, 4 heads, d_mlp 512,
  ReLU, no LayerNorm, learned positional embeddings, untied embed/unembed,
  Gaussian init scaled by 1/sqrt(fan_in). Unembedding outputs logits over the
  97 answers only (not the full 99-token vocab) — the `op`/`=` tokens can never
  be answers, so there is nothing to learn there.
- Deviations from Nanda: p = 97 (not 113); 4 input tokens `[a, op, b, =]` (Nanda
  used 3, `[a, b, =]`); AdamW weight decay applies to *all* parameters
  (same as Nanda's code).
- Speed: the first naive implementation ran at ~34 ms/step (CPU) / ~28 ms/step
  (MPS). Since only the `=` position feeds the logits, the (last) block now
  computes attention queries and the MLP for that position only. Checked this
  is exact (max abs diff 0.0 vs. the full computation on a 2-layer model).
  Now ~13 ms/step on both CPU and MPS.
- Device pinned to CPU in the config (auto-select CUDA → MPS → CPU still
  available via `device=auto`): same speed as MPS here, supports the float64
  loss, and is bit-for-bit reproducible.

## 2026-10-30 — Phase 0: make it grok

### Attempt 1 — starting hyperparameters (grokked first time)
Config = the `baseline` defaults: p 97, wd 1.0, train_frac 0.3 (2 823 train /
6 586 test pairs), 25 000 steps, 1 layer, d_model 128 / 4 heads / d_mlp 512,
AdamW lr 1e-3, betas (0.9, 0.98), 10-step linear warmup, full batch, float64
loss, seed 0, CPU. (Ran before the determinism fix below.)

- train ≥99% at step **132**; test still 1.7% at step 139.
- test ≥50% at step **9 507**, ≥95% at step **10 525**; final train 100%, test 100%.
- 314 s wall-clock.
- Interpretation: textbook grokking on the first try. Memorisation in ~130 steps,
  then ~9 000 steps where test accuracy creeps 2% → 15%, then a sharp jump to
  ~100% over ~1 500 steps. No hyperparameter changes were needed; following
  Nanda's setup (no LayerNorm, short warmup, untied embeddings, wd 1.0) worked
  as the brief predicted.

### Fix — runs were not bit-for-bit reproducible
Two identical 300-step runs differed. Gradients of every parameter except W_E
matched exactly; W_E differed by ~9e-9 on a single backward pass. Cause: the
embedding lookup's backward (`index_put_` with accumulate) is nondeterministic
on multithreaded CPU. Fixed with `torch.use_deterministic_algorithms(True)` in
`set_seed`; two runs now give identical metrics and byte-identical embedding
snapshots. Attempt 1's numbers above are therefore one sample of a slightly
noisy process; everything from here on is exactly regenerable.

### Attempt 2 — same config, deterministic (this is `baseline`)
- train ≥99% at step **132**, test ≥50% at **9 035**, ≥95% at **10 003**,
  final 100% / 100%. 260 s.
- The ~1e-8 nondeterminism moved the grokking step by ~500 steps (5%) — a
  reminder that the exact step is sensitive and shouldn't be over-interpreted.

### Fix — recording schedule hit 160 steps / 38 snapshots, not ~200 / ~40
`np.round(np.logspace(...))` collapses many early points onto the same integer,
so deduplication left only 160 steps. `log_steps` / `snapshot_steps` now grow
the raw count until exactly 200 / 40 distinct steps survive. Re-ran baseline:
recording is no-grad and consumes no RNG, so training is unaffected (identical
final losses, 8.59e-08 train / 1.95e-07 test).

### Final baseline (attempt 2 re-recorded) — Phase 0 result
| milestone | step |
|---|---|
| train ≥ 99% | 130 |
| test ≥ 50% | 8 939 |
| test ≥ 95% | 10 065 |
| final train / test acc | 100% / 100% |

Test acc at a few recorded steps: 4.2% @ 1 098, 11.3% @ 5 785, 21.0% @ 7 631,
93.5% @ 9 675. Chance is 1/97 ≈ 1.0%. Plot: `notes/plots/phase0_baseline.png`.

Other observations:
- **Test loss rises** from 4.6 at init to ~28 around step 1–2k while train loss
  falls to ~3e-7 — the model is confidently wrong on unseen pairs during the
  memorisation phase — then collapses to ~2e-7 at grokking.
- **Train loss plateaus at ~2–3e-7** from step ~1 000 to ~8 000 and drops again
  (to ~9e-8) right as test accuracy jumps: the internal reorganisation is visible
  in train loss too, even though train accuracy was already 100%.
- **Test accuracy isn't flat** before the jump: it creeps up steadily on the log
  axis (2% → 21% between steps 150 and 7 600). Consistent with Nanda et al.'s
  "circuit formation" phase happening gradually before the visible cleanup.

**Phase 0 took 1 attempt** (plus a deterministic re-run and a re-record; no
hyperparameter changes).

## 2026-10-30 — Phase 1: sweep and export

### Bug — every embedding snapshot was the final embedding
After projecting the baseline, every frame (step 1 included) was identical
to the final one. `number_embeddings()` returned `W_E[:p].detach()`, which is
a *view* of the live parameter; on CPU `.float().cpu().numpy()` are no-ops,
so all 40 stored "snapshots" pointed at the same memory and the whole stack
showed the final weights. The metrics were unaffected. Fix: `.clone()`. Also
added a `validate.py` check that the first and final frames differ, so this
can't silently come back. The Phase 0 baseline metrics are unchanged (training
doesn't depend on recording); all four runs were re-run with the fix.
Lesson: look at the data, not just at whether the pipeline ran.

### The four sweep runs (all seed 0, 25 000 steps, deterministic CPU)
All four were trained in parallel (4 processes on the M3, ~7–16 min each under
contention vs ~4.5 min alone). The baseline reproduced its Phase 0 numbers
exactly, so parallel execution doesn't affect results.

| run | diff from baseline | train ≥99% | test ≥50% | test ≥95% | final train / test | max test |
|---|---|---|---|---|---|---|
| `baseline` | — | 130 | 8 939 | 10 065 | 100% / 100% | 100% |
| `no_decay` | wd 1.0 → **0.0** | 125 | never | never | 100% / **3.0%** | 3.05% |
| `more_data` | frac 0.3 → **0.6** | 130 | **125** | **378** | 100% / 100% | 100% |
| `less_data` | frac 0.3 → **0.15** | 111 | never | never | 100% / **0.5%** | 1.04% |

Every run behaved as the brief hypothesised, so no labels or values needed
changing.

- **no_decay**: memorises at the same speed as baseline, then *nothing happens*
  for 25k steps. Test acc creeps from ~2.8% at step 1k to 3.0% and stops; train
  loss falls to 5.6e-11 (no decay pulling the weights back), test loss rises to
  56.7. This is the headline contrast: same data, same steps, the only
  difference is weight decay, and generalisation never comes.
- **more_data**: no meaningful delay. Test acc is already 62% at the step train
  reaches 99% (130), and 95% at step 378. Generalisation and memorisation
  essentially coincide, i.e. no grokking gap.
- **less_data** (1 411 training pairs): memorises fastest (step 111, fewer
  examples) and stays at chance (max 1.04%, final 0.5%; chance = 1.03%) for all
  25k steps. It may still grok much later; we don't know, and the site must not
  imply it never would, only that it hadn't by 25k steps.

Embeddings (fixed-basis PCA, fitted on each run's final snapshot):
- baseline: random blob until ~7.6k, then organises into concentric
  ring/ellipse structure by 12k. 2 PCs explain 26.8% of final variance.
- more_data: rings already visible by step ~254 and stable afterwards (23.8%).
- no_decay / less_data: blobs throughout; the 2 PCs explain only 8.3% / 16.1%,
  i.e. no dominant low-dimensional structure.
- **Surprise**: colouring by numeric value does *not* give a smooth colour
  wheel around the rings; neighbours on a ring aren't consecutive numbers.
  That's consistent with Nanda et al.: the ring orders numbers by k·a mod p for
  some key frequency k, not by a. (Investigating which k: out of scope → next
  steps.)

Export: 4 files of ~70 KB each (limit 200 KB), `validate.py` passes. Spot-
checked `no_decay.json`: 200 steps from 1 to 25 000, full config present,
test acc tail 0.0302, 40 frames of 97 × 2 coords.

## 2026-10-30 — Phases 2–3: the site

No training runs in these phases. Notes for the writeup:
- The headless check (`check_site.py`) caught a bug *in the check itself*:
  after clicking a run button, it read the readout before the new run had
  loaded, so each "pass" was actually showing the previous run's numbers. The
  app now sets `body[data-run]` once a run has rendered; the check waits for
  it and compares the readout against the run's own JSON.
- Explanation-text numbers are static HTML (easy to edit), but every number is
  tagged `data-fact="…"` and `check_site.py` recomputes each one from
  `site/data/` and fails on any mismatch, so the text can't silently drift
  from the data.
- Initial view transfers ~150 KB uncompressed (page, CSS, JS, manifest,
  baseline run, Chart.js from the CDN); other runs load only when selected.

---

# Summary (for the writeup)

**Goal.** Reproduce grokking (Power et al. 2022) on `(a + b) mod 97` with a
1-layer transformer following Nanda et al. 2023, and build an interactive
page that shows it, plus the effect of weight decay and dataset size.

**Phase 0 took 1 attempt.** The starting hyperparameters (wd 1.0, 30% data,
lr 1e-3, betas (0.9, 0.98), full batch, no LayerNorm, 10-step warmup) grokked
first time. Nothing failed for hyperparameter reasons. The failures were
engineering bugs:
1. **Nondeterminism.** CPU embedding-gather backward differed by ~1e-8 between
   identical runs; fixed with `torch.use_deterministic_algorithms(True)`. That
   tiny noise moved the grokking step by ~5% (test 95% at 10 525 vs 10 003).
2. **Recording schedule.** Rounding log-spaced steps to integers left 160
   distinct steps instead of 200 (and 38 snapshots instead of 40). Fixed by
   growing the raw count until enough distinct steps survive.
3. **Snapshot aliasing.** Every embedding "snapshot" was a view of the live
   weights, so all frames showed the final state. Caught by plotting, fixed
   with `.clone()`, now guarded by `validate.py`.
4. **A racy test.** The headless check read the readout before a newly
   selected run had loaded, so it "passed" while looking at the previous run.
   Fixed by waiting on an explicit render signal and comparing against the JSON.

**Final outcome per run** (seed 0, 25 000 steps; first recorded step at threshold):

| run | train ≥99% | test ≥50% | test ≥95% | final test acc | as hypothesised? |
|---|---|---|---|---|---|
| baseline (wd 1.0, 30%) | 130 | 8 939 | 10 065 | 100% | yes: groks, ~77× delay |
| no_decay (wd 0, 30%) | 125 | never | never | 3.0% (max 3.05%) | yes: never generalises |
| more_data (wd 1.0, 60%) | 130 | 125 | 378 | 100% | yes: essentially no delay |
| less_data (wd 1.0, 15%) | 111 | never | never | 0.5% (max 1.04%) | yes, within the 25k budget |

**Surprising / worth mentioning.**
- How *easy* it was: the published recipe worked unmodified at p = 97 with
  4-token inputs.
- Test accuracy isn't flat before the jump. On a log axis it creeps from ~2% to
  ~21% over steps 150–7 600, then jumps to 93% by 9 675. Train loss also
  plateaus (~3e-7) and then drops again exactly at the transition, so the
  reorganisation is visible even on the training set.
- During memorisation, test loss *rises* to ~28 (confidently wrong), then
  collapses to ~2e-7.
- Without weight decay, train loss reaches 5.6e-11 and test loss 56.7: the
  network keeps sharpening its memorised answers instead of simplifying.
- Embeddings: the grokked runs form concentric rings in their top-2 PCs, but
  the rings aren't ordered by number value. Non-grokked runs stay as clouds,
  and their top-2 PCs explain much less variance (8% / 16% vs 27% / 24%).
- A tiny numerical perturbation (1e-8) shifted the grokking step by 5%.
  Single-seed step counts should be quoted as approximate.

**Next steps** (all deliberately out of scope here):
- Multiple seeds per configuration, to put error bars on the grokking step.
- Run `less_data` for 100k+ steps: delayed or never?
- Sweep weight decay (e.g. 0, 0.03, 0.1, 0.3, 1, 3) to find where grokking
  turns on and how the delay scales.
- Fourier analysis of the embeddings (which key frequencies; does the ring
  order follow k·a mod p?), following Nanda et al.
- Other operations (subtraction, division mod p), and denser snapshots around
  the transition for a smoother embedding animation.

## 2026-10-30 — Final reproducibility check
Moved `outputs/` aside and ran `./run_all.sh` from scratch (sequential, 1 180 s
total: less_data 132 s, baseline 257 s, no_decay 256 s, more_data 526 s).
Afterwards `git status` was clean: every exported JSON and both diagnostic
plots were regenerated byte-for-byte.
