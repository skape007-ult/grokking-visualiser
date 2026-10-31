# Grokking Visualiser

An interactive page that lets you watch a tiny neural network suddenly
*generalise* on modular arithmetic, long after it has memorised its training
data ("grokking"), and see that removing weight decay stops it happening.

## Phase 0 — Make it grok

**What was built.** A config-driven training pipeline (`train/`) for a
1-layer transformer on `(a + b) mod 97`, and a baseline run that reproduces
grokking: train accuracy hits 99% by step 130, test accuracy stays near chance
for thousands of steps, then jumps to 95% at step ~10 000
(`notes/plots/phase0_baseline.png`). It worked on the first attempt — see
`notes/log.md`.

**Key decisions and why.**
- *Follow Nanda et al. 2023 closely* (no LayerNorm, learned positional
  embeddings, untied embed/unembed, AdamW lr 1e-3, wd 1.0, betas (0.9, 0.98),
  10-step warmup, full batch, 30% data) because it is the most reliable known
  recipe. Deviations: p = 97 instead of 113, and inputs are 4 tokens
  `[a, op, b, =]` instead of 3.
- *Logits over the 97 answers only*, not the full 99-token vocabulary: `op` and
  `=` can never be answers, so there is nothing to learn there.
- *Only the `=` position is computed in the last block.* Only that position's
  output feeds the logits, so this is exact (verified: max diff 0.0) and makes a
  step ~2.6× cheaper (~13 ms on an Apple M3).
- *CPU by default, deterministic algorithms on.* On the M3 CPU and MPS are the
  same speed here; CPU supports the float64 loss, and with
  `torch.use_deterministic_algorithms(True)` a run is bit-for-bit regenerable
  from its config (the embedding backward pass is otherwise nondeterministic).
  `device=auto` selects CUDA → MPS → CPU if you prefer.
- *Every run is a `RunConfig`* (`train/config.py`); exploratory variants use
  `--set key=value` rather than edited constants, and the full config is saved
  next to every result.

