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

**Final outcome per run**: three seeds each (0 / 1 / 2), 25 000 steps, first
recorded step at threshold, with adaptive recording (see the 2026-10-31 entry).
The original seed-0-only table is in the Phase 1 entry.

| run | train ≥99% | test ≥50% | test ≥95% | final test acc | as hypothesised? |
|---|---|---|---|---|---|
| baseline (wd 1.0, 30%) | 130 / 130 / 130 | 8 939 / 4 563 / 3 744 | 9 868 / 5 785 / 4 216 | 100% ×3 | yes, all 3 grok; delay 32–76× |
| no_decay (wd 0, 30%) | 125 / 120 / 120 | never ×3 | never ×3 | 3.0% / 4.1% / 4.3% | yes, none generalise |
| more_data (wd 1.0, 60%) | 128 / 128 / 138 | 124 / 120 / 133 | 370 / 269 / 489 | 100% ×3 | yes, essentially no delay |
| less_data (wd 1.0, 15%) | 111 / 111 / 111 | never ×3 | never ×3 | 0.5% / 0.4% / 1.1% (max 1.6%) | yes, within the 25k budget |

**Surprising / worth mentioning.**
- **Seed 0 was the slowest of three to grok** (9 868 vs 5 785 and 4 216). The
  single-seed "77× later" headline overstated the typical delay; across seeds
  it's 32–76×. Same qualitative story, very different number.
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
- 10–20 seeds per configuration, to characterise the grokking-step
  distribution (three seeds already span 4 216–9 868).
- Run `less_data` for 100k+ steps: delayed or never?
- Sweep weight decay (e.g. 0, 0.03, 0.1, 0.3, 1, 3) to find where grokking
  turns on and how the delay scales.
- Fourier analysis of the embeddings (which key frequencies; does the ring
  order follow k·a mod p?), following Nanda et al.
- Other operations (subtraction, division mod p), other primes, deeper models.

## 2026-10-30 — Final reproducibility check
Moved `outputs/` aside and ran `./run_all.sh` from scratch (sequential, 1 180 s
total: less_data 132 s, baseline 257 s, no_decay 256 s, more_data 526 s).
Afterwards `git status` was clean: every exported JSON and both diagnostic
plots were regenerated byte-for-byte.

## 2026-10-30 — Review pass
A full review of the code, plus checks in WebKit and Firefox as well as
Chromium. The training code needed no changes. Fixes were all on the site,
export and deploy side:
- **Mobile scroll trap:** `touch-action: none` on the embedding canvas meant a
  swipe starting on the plot couldn't scroll the page on phones (the plot is
  ~324 px of a 390 px screen). Changed to `manipulation`; pinning a number now
  uses `click` instead of `pointerdown`.
- **Run load failure:** a failed fetch after a button click was unhandled. The
  new button stayed pressed over the old run's data and no message appeared.
  Now an error is shown, the toggle reverts, and a retry works.
- **Accessibility:** removed `aria-live` from the readout, which made screen
  readers repeat the slider's `aria-valuetext` on every drag tick.
- **Efficiency:** CSS variables and the 97 viridis colours are now computed
  once instead of on every slider move.
- **Export:** `export.py` now builds everything in memory first, so a missing
  run aborts before `site/data/` is touched. Output is byte-identical to before.
- **Pages workflow:** `actions/deploy-pages` v4+ needs `permissions: actions:
  read`, which was missing and might have failed the first deploy. Also bumped
  to the current majors (checkout v7, configure-pages v6, upload-pages-artifact
  v5, deploy-pages v5); none of their breaking changes affect this repo.
- **Checks:** `check_site.py` gained `--browsers` (Chromium, WebKit with the
  iPhone 13 profile, Firefox) and regression checks for the bugs above. Run
  against the pre-fix code, 3 checks fail; on the fixed code all pass in all
  three engines. Dev dependencies are pinned in `requirements-dev.txt`.
- **Docs:** `ARCHITECTURE.md` is a function-by-function map of the pipeline and
  the data contract. The README links to it, and its duplicated
  reproduce/deploy commands were merged into one place.
- `run_all.sh` gained `JOBS=N` for parallel training (measured 1 051 s vs 1 180 s
  sequential, ~11% faster; a from-scratch `JOBS=4` run regenerated `site/data/`
  and the plots byte-for-byte). Only ~11% because the 4 runs contend for the
  M3's 4 performance cores.

## 2026-10-31 — Second review pass: seeds and transition frames

Prompted by an external review: the central claim (weight decay is what makes
generalisation happen) rested on n = 1 per condition.

**Recording change (made first, since seed 0 had to be re-run anyway).**
In the old baseline only 2 embedding frames covered test accuracy 21% → 94%
(steps 7 631 and 9 675), so the rings appeared to snap in. New config fields:
- `dense_trigger_rate = 0.2`: when |Δtest_acc| per e-fold of steps exceeds
  this, the next record comes 2% of steps later (`dense_ratio`) instead of
  the base ~5%.
- `snapshot_delta = 0.05`: snapshot whenever test accuracy has moved 5 points
  since the last snapshot.

Recording reads metrics only and never touches training, so the trajectories
are unchanged. Seed 0's final losses are identical to before. The
first-crossing numbers moved slightly because the transition is sampled more
finely: baseline test ≥95% at 9 868 (was 10 065), more_data 370 (was 378),
more_data train ≥99% at 128 (was 130). The baseline now has 10 frames between
21% and 94% test accuracy. File sizes are 77–98 KB (limit 200 KB).
no_decay and less_data never trigger dense recording, since their test
accuracy barely moves.

**Seeds.** `SEEDS = (0, 1, 2)` for every condition: 12 runs, trained 4 at a
time in 3 261 s (54 min) on the M3. Results (seed 0 / 1 / 2):
- baseline: groks in all three. Test ≥95% at **9 868 / 5 785 / 4 216**, a 2.3×
  spread. Train ≥99% at 130 in all three.
- no_decay: **never generalises in any seed.** Final test 3.0% / 4.1% / 4.3%,
  max 4.4%.
- more_data: 95% at 370 / 269 / 489. Essentially no delay in all three.
- less_data: chance in all three. Final 0.5% / 0.4% / 1.1%, max 1.6%.

**Interpretation.** The qualitative claims are robust to seed: weight decay is
needed (0/3 without, 3/3 with), more data removes the delay, and less data
stalls. The *timing* is not robust. Seed 0 happened to be the slowest, so the
old "77×" figure overstated the typical delay. The site and README now quote
ranges, and `check_site.py` verifies them against all seeds.

**Other changes in this pass:**
- **Site:** faint lines for seeds 1–2, and a log-uniform slider (400 positions
  mapped to the nearest recorded step), so dense recording doesn't make the
  slider uneven. The embedding canvas `aria-label` now describes the current
  frame.
- **Limitations:** a section in the README and a paragraph on the page.
- **LayerNorm:** its omission is now stated as deliberate (following Nanda,
  for interpretability).
- **Licence and run list:** MIT `LICENSE`. `validate.py` takes the run order
  from `config.RUN_ORDER` (no hard-coded list).
- **Checks:** one of my new checks was itself wrong. The "timing varies more
  than 2×" check divided the slowest *other* seed by the fastest seed overall
  (1.37×). It now uses all three (2.34×).
- **Zsh pitfall** while doing this by hand: `$R` with several ids isn't
  word-split in zsh. `run_all.sh` is bash, so it's unaffected.
- `requirements.txt` was already pinned exactly (torch 2.15.0, numpy 2.5.4,
  matplotlib 3.11.2), so no change was needed there.

