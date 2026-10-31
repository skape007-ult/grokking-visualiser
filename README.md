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

## Phase 1 — Sweep and export

**What was built.** Four runs that differ from `baseline` in exactly one
variable, a fixed-basis PCA projection of the embeddings, and an exporter +
validator for the data the site consumes (`site/data/`).

| run (UI label) | change | outcome (25 000 steps) |
|---|---|---|
| `baseline` ("Standard training") | — | groks: test acc 95% at step ~10 000 |
| `no_decay` ("No weight decay") | weight decay 0 | memorises, test acc stuck at 3% |
| `more_data` ("More training data") | 60% of pairs | generalises almost immediately (95% at step 378) |
| `less_data` ("Less training data") | 15% of pairs | memorises, test acc at chance (~1%) |

Diagnostic plot: `notes/plots/phase1_sweep.png`.

**Key decisions and why.**
- *200 log-spaced recorded steps, 40 embedding snapshots.* Grokking spans
  four orders of magnitude of steps, so linear spacing would waste almost every
  point on the flat tail.
- *PCA basis fitted once per run, on the final snapshot,* and every frame is
  projected onto it. Fitting per frame would let the axes rotate between frames,
  making the animation meaningless. Component signs are fixed deterministically.
- *Raw outputs live in gitignored `outputs/`; only `site/data/` is committed,*
  and only `train/export.py` writes it. `train/validate.py` enforces the data
  contract (lengths, ranges, rounding, frame steps ⊂ steps, file sizes,
  manifest ↔ files) and that first and final frames differ, which catches the
  snapshot-aliasing bug described in `notes/log.md`.

**Reproduce** (from the repo root, ~5 min per run alone on an Apple M3 CPU):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train/train.py --run baseline     # likewise: no_decay, more_data, less_data
.venv/bin/python train/project.py baseline no_decay more_data less_data
.venv/bin/python train/export.py
.venv/bin/python train/validate.py
```

or simply `./run_all.sh` (also regenerates the plots in `notes/plots/`).

## Phase 2 — The curves, online

**What was built.** `site/`: a static page (plain HTML + vanilla JS +
Chart.js 4.4.1 from jsDelivr, no build step) showing train and test
accuracy on a log-step axis, with a slider that moves a marker along the
chart and updates a readout ("Step 9,675 — train 100%, test 94%"). A
GitHub Actions workflow publishes `site/` to GitHub Pages.

**Key decisions and why.**
- *The slider indexes recorded steps, not raw step numbers,* so it moves
  evenly along the log axis (otherwise the first 1 000 steps, where
  memorisation happens, would be squeezed into 4% of the track).
- *Relative fetch paths* (`data/manifest.json`), so the same files work from
  `python -m http.server` and under the Pages subpath `/<repo>/`.
- *Chart animations off;* moving the slider only redraws the marker
  (`chart.draw()`), with no data update, so scrubbing feels instant.
- *Headless check* (`check_site.py`, Playwright) serves `site/`, then at
  desktop and phone sizes asserts no console errors or failed requests, that
  the chart renders, that keyboard, mouse-drag and touch move the readout, and
  that there's no horizontal scroll on mobile. Playwright is a dev-only
  dependency, not in `requirements.txt`.

**Preview locally**

```bash
cd site && python3 -m http.server 8000     # then open http://localhost:8000
```

(Opening `index.html` directly from disk won't work: browsers block `fetch`
on `file://`. The page shows an error saying so.)

**Run the headless check** (optional)

```bash
.venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium
.venv/bin/python check_site.py --screenshots
```

**Deploy to GitHub Pages**
1. Create an empty repo on GitHub (e.g. `grokking-visualiser`), with no README.
2. `git remote add origin git@github.com:<you>/grokking-visualiser.git && git push -u origin main`
3. On GitHub: **Settings → Pages → Build and deployment → Source: "GitHub Actions"**.
4. Re-run the "Deploy site to GitHub Pages" workflow from the **Actions** tab
   (the first push may have run before Pages was enabled). The site goes live at
   `https://<you>.github.io/grokking-visualiser/`.

## Phase 3 — Embeddings and the run toggle

**What was built.**
- *Embedding panel:* the 97 number embeddings in each run's fixed 2-D PCA
  basis, driven by the same slider (it shows the latest snapshot at or
  before the slider's step). Fixed, equal-aspect axis limits per run; viridis
  colouring by value with a legend; hover (or tap) shows the number, and a
  click/tap pins it so you can follow one number through training.
  Hand-drawn `<canvas>`, sized to the device pixel ratio so it stays crisp.
- *Run toggle:* four real `<button>`s with `aria-pressed`, built from the
  manifest. Switching keeps the slider at the same step, refits the embedding
  axes, lazy-loads and caches the run, and writes `#run=<id>` to the URL.
  Deep links like `#run=no_decay` work.
- *Explanation:* three paragraphs plus references, with every number taken
  from the exported data (and checked by `check_site.py`).
- *Mobile:* single column under 720px; the slider + readout bar is sticky at
  the bottom of the screen while the visualisation is in view, so you can
  scrub while looking at either panel. Touch targets ≥ 44px.

**Key decisions and why.**
- *Show the latest embedding snapshot at or before the slider's step,* never
  a later one: the panel should never show the future.
- *Equal x/y scale,* because distances in PCA space are meaningful; stretching
  one axis would distort the rings.
- *Static explanation text, checked against the data,* rather than text
  generated in JS: easy for a human to edit, still impossible to go stale
  silently.

Screenshots (from `check_site.py --screenshots`): `notes/screenshots/`.

