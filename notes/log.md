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

