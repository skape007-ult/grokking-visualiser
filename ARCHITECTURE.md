# How it works

This is a code-level map of the project: what each stage does, what each file
and function is for, and the contract between the stages. For *what was
found* and *why* the main design decisions were made, see the
[README](README.md) (including its build-notes appendix). The dated history
of every run and bug is in [`notes/log.md`](notes/log.md).

## The pipeline

The project is two halves that only meet through JSON files:

```
            train/  (Python, PyTorch)                         site/  (static HTML + JS)
 ┌──────────────────────────────────────────────┐        ┌──────────────────────────────┐
 │ config.py ─► train.py ─► outputs/<id>/       │        │ index.html                    │
 │   RunConfig   │  data.py   run.json          │        │ app.js  ◄── fetch ──┐         │
 │   SWEEP       │  model.py  embeddings.npy    │        │ style.css            │         │
 │               ▼                              │        └──────────────────────┼────────┘
 │           project.py ─► outputs/<id>/projected.json                          │
 │               ▼                                                              │
 │           export.py ──────────────► site/data/manifest.json, runs/<id>.json ─┘
 │               ▼                              │
 │           validate.py (checks site/data)     │        check_site.py (drives the site
 │           plot.py (diagnostic PNGs)          │        in real browsers)
 └──────────────────────────────────────────────┘
```

- `train/` never imports anything from `site/`, and `site/` never runs Python.
  The only coupling is the [data contract](#the-data-contract) below.
- `outputs/` (gitignored) holds raw, full-precision results. `site/data/`
  (committed) holds the rounded, contract-shaped copy the page reads.
- Only `export.py` writes `site/data/`. Nothing is ever hand-edited there.
- `run_all.sh` runs the whole left half in order.

## Stage by stage

### 1. Configuration — `train/config.py`

- **`RunConfig`** (frozen dataclass): every setting that affects a run
  (task, model size, optimiser, recording density, seed, device). Nothing that
  changes results is hard-coded elsewhere.
  - `to_dict()`: the full config, saved next to every result.
  - `with_overrides(**kw)`: a copy with some fields changed (used to derive
    sweep variants).
- **`RunSpec`**: a `RunConfig` plus how the site presents it (`label`,
  `description`).
- **`SWEEP`**: the four runs, in display order. Each differs from `BASELINE` in
  exactly one field. **`RUNS`** indexes them by id.
- **`parse_override(cfg, "key=value")`**: applies a command-line `--set`,
  casting the value to the field's type (used for exploratory attempts).

### 2. Data — `train/data.py`

- **`make_dataset(cfg)`** builds all p² = 9,409 inputs `[a, op, b, =]`
  (token ids: numbers `0…p-1`, `op = p`, `= = p+1`) with labels `(a + b) mod p`.
  It shuffles them with a NumPy generator seeded from `cfg.seed` and splits off
  `train_frac` for training. Returns CPU tensors
  `(train_x, train_y, test_x, test_y)`.

### 3. Model — `train/model.py`

- **`Transformer`**: token embedding `W_E` (p+2 × d_model) + learned positional
  embedding `W_pos`, `n_layers` × `Block`, then an unembedding `W_U`
  (d_model × p) that reads only the final (`=`) position. Logits cover the p
  possible answers only.
  - `number_embeddings(p)`: a **copy** (`.clone()`) of the first p rows of
    `W_E`, the vectors plotted on the site. A plain slice would be a live
    view, and every snapshot would silently show the final weights; this
    happened once and is documented in the log.
- **`Block`**: multi-head causal self-attention (no LayerNorm) + ReLU MLP, each
  with a residual connection. With `last_only=True` (used for the last block)
  it computes queries and the MLP for the final position only. This is exact,
  because only that position feeds the logits, and it is ~2.6× faster.
- Initialisation: Gaussian scaled by 1/√fan_in, zero biases (as in Nanda et al.).

### 4. Training — `train/train.py`

`python train/train.py --run <id> [--set key=value …] [--tag name]`

- **`set_seed(seed)`** seeds Python, NumPy and torch (and CUDA if present), and
  turns on `torch.use_deterministic_algorithms`. Together with the CPU device,
  this makes a run bit-for-bit repeatable.
- **`pick_device(name)`**: `"auto"` → CUDA → MPS → CPU. The config defaults to
  `"cpu"`.
- **`log_steps(final, n)`**: exactly n distinct, log-spaced integer steps from
  1 to `final`. This is where metrics are recorded.
- **`snapshot_steps(record, n)`**: n of those steps, also log-spaced, where the
  embedding matrix is saved. Always includes the first and final step.
- **`evaluate(model, x, y, dtype)`**: loss and accuracy on a full split, no grad.
- **`train(cfg, out_name)`**: full-batch AdamW with a linear warmup, for
  `cfg.steps` steps. At each recorded step it evaluates on the full train and
  test sets, and at snapshot steps it stores the number embeddings. It writes
  `outputs/<name>/run.json` (config, runtime info, metric arrays, snapshot
  steps) and `embeddings.npy` (snapshots × p × d_model).
  - The loss is computed in float64, or float32 on MPS, which has no float64.
    The dtype actually used is recorded.

### 5. Projection — `train/project.py`

`python train/project.py <id> …`

- **`fit_pca(X, k=2)`**: mean + top-k principal directions via SVD. Each
  component's sign is flipped so that its largest loading is positive, so
  re-runs never mirror the plot.
- **`project_run(id)`** fits the PCA **once, on the final snapshot**, and
  projects every snapshot with that same basis. The axes therefore stay fixed
  through the animation, and motion in the plot is motion of the embeddings.
  Writes `outputs/<id>/projected.json` (snapshot steps, `[frames][p][2]`
  coordinates, variance explained by the two components).

### 6. Export — `train/export.py`

`python train/export.py`

- **`r(x)`** rounds floats (recursively) to 4 decimal places and rejects
  NaN/Inf.
- **`build()`** reads every run in `SWEEP` from `outputs/`. It refuses any run
  whose saved config differs from the one in `config.py` (stale outputs), and
  assembles the manifest and per-run JSON in memory.
- **`export()`** calls `build()` first, so a missing or stale run aborts
  before anything is written. It then replaces `site/data/runs/` and writes
  `manifest.json`. Run files are compact (~70 KB each, ~24 KB gzipped).

### 7. Validation — `train/validate.py`

`python train/validate.py` (exits non-zero on any failure)

It checks every rule of the data contract:
- **Manifest:** run order, required fields, and files that exist with no
  orphans. Each run's config agrees with its manifest entry.
- **Steps:** integer and strictly increasing, from 1 to the final step.
- **Metrics:**
  - every array is the same length as `steps`
  - accuracies are in [0, 1] and losses are ≥ 0
  - no NaN/Inf, and values are rounded to 4 dp
- **Frames:** sorted, a subset of `steps`, and include the first and final
  step. Each has exactly p `[x, y]` pairs.
- **Size and sanity:** each file is ≤ 200 KB, and the first and final frames
  differ (this catches the snapshot-aliasing bug).

### 8. Diagnostics — `train/plot.py`

- `summary <dirs…>`: milestone steps per run (train ≥ 99%, test ≥ 50% / 95%,
  final metrics), as used in the log.
- `curve <dir> <png>`: accuracy and loss curves for one run
  (`notes/plots/phase0_baseline.png`).
- `sweep <png> <dirs…>`: side-by-side accuracy curves for all runs
  (`notes/plots/phase1_sweep.png`).

## The data contract

`site/data/manifest.json`:

```json
{
  "p": 97,
  "operation": "(a + b) mod p",
  "runs": [
    { "id": "baseline", "label": "Standard training",
      "description": "Weight decay 1.0, 30% of data",
      "weight_decay": 1.0, "train_frac": 0.3, "file": "runs/baseline.json" }
  ]
}
```

Runs are listed in display order: `baseline`, `no_decay`, `more_data`,
`less_data`. `label` is the meaning-based button text, and `description` is
the config-style subtitle.

`site/data/runs/<id>.json`:

| field | shape | meaning |
|---|---|---|
| `config` | object | the full `RunConfig` that produced the run |
| `steps` | `[n]` ints | recorded steps, 1 … final, strictly increasing (n = 200) |
| `train_acc`, `test_acc` | `[n]` floats in [0, 1] | accuracy on the full split at each step |
| `train_loss`, `test_loss` | `[n]` floats ≥ 0 | cross-entropy at each step |
| `embedding_frames` | `[m]` of `{step, coords}` | m = 40 frames; `step` ∈ `steps`; `coords[i]` = 2-D position of number i (p entries) |

All floats are rounded to 4 dp. `validate.py` enforces all of this.

## The page — `site/`

### `index.html`

Static markup:
- the header and lede
- an empty `#run-toggle` group, filled from the manifest
- the two panels: the `#curve` canvas for Chart.js and the `#embedding` canvas
- the sticky slider + readout bar, and an error banner
- the explanation text, where every number is wrapped in
  `<span data-fact="…">` so `check_site.py` can verify it against the data

Chart.js 4.4.1 is loaded from jsDelivr with an SRI hash.

### `app.js`

One `state` object (`manifest`, current `runId` / `run`, slider `index`,
embedding axis limits, hovered and pinned number) and plain functions:

**Data loading**
- `fetchJSON(path)`: fetches with a relative path, so the page works under a
  GitHub Pages subpath.
- `loadRun(id)`: lazily fetches a run and caches the promise. A failed load is
  evicted from the cache so it can be retried.

**Accuracy chart**
- `buildCurveChart()`: the Chart.js line chart, with animations off and a
  log x-axis labelled 1, 10, 100, 1k, 10k (`logTick`).
- `markerPlugin`: draws the vertical step marker.
- `showRunOnChart(run)`: swaps in a run's data.

**Embedding plot**
- `viridis(t)`: the colour ramp, interpolated from 11 stops. `pointColors`
  caches one colour per number.
- `embeddingLimits(run)`: square, fixed axis limits over every frame of the
  run, plus a 6% margin.
- `frameIndexFor(run, step)`: the latest frame at or before `step`.
- `drawEmbedding()`: renders the frame on a canvas sized to
  `devicePixelRatio`, plus a ring around hovered or pinned points.
- `nearestPoint(evt)` and `initEmbeddingInteraction()`: hover shows a tooltip,
  and a click or tap pins a number. Pinning uses `click`, not `pointerdown`,
  so swipes still scroll the page on phones. A `ResizeObserver` keeps the
  canvas crisp and square.

**Slider and readout**
- `setIndex(i)`: moves the slider and marker, then updates the readout text,
  `aria-valuetext` and the embedding frame.
- `nearestIndex(run, step)`: finds the equivalent position when switching runs.

**Run toggle**
- `buildToggle()`: creates one `<button aria-pressed>` per manifest run.
- `runFromHash()`: reads `#run=<id>`, falling back to the first run.
- `selectRun(id)`: loads the run and re-renders both panels, keeping the
  slider at the same step.
  - A token guards against out-of-order loads.
  - If a load fails, the pressed state reverts and the current run stays
    usable.
  - It sets `body[data-run]` once a run has rendered (used by the tests).

**Boot**
- `showError(err)`: shows the error banner.
- `main()`: loads the manifest, builds the chart, toggle and embedding
  interaction, wires up the slider and `hashchange`, then selects the run named
  in the URL.

### `style.css`

- Colours are CSS custom properties.
- A two-column panel grid collapses to one column under 720px.
- The run toggle is a 4-up segmented grid (2×2 under 900px).
- The slider bar is sticky at the bottom of the visualisation, so it stays
  reachable on phones.
- Touch targets are ≥ 44px.

## Verification — `check_site.py`

`python check_site.py [--browsers chromium,webkit,firefox|all] [--screenshots]`

It serves `site/` on a free local port and drives it in headless browsers via
Playwright:

| check | what it asserts |
|---|---|
| `check_facts` | every `data-fact` number in `index.html` matches a value recomputed from `site/data/` |
| `check_common` | no console errors or failed requests; chart has data; embedding canvas is drawn, square and high-DPI |
| `check_slider` | Home / arrow keys / mouse drag move the readout; `aria-valuetext` mirrors it; the embedding changes and never shows a later frame |
| `check_toggle` | buttons come from the manifest; each click sets `aria-pressed` and the hash, keeps the step, and the readout matches that run's JSON |
| `check_desktop` | the above, plus lazy loading, hover tooltip and click-to-pin |
| `check_load_failure` | a blocked run file shows an error, reverts the toggle, keeps the current data, and a retry works |
| `check_hash` | `#run=…` deep links, live hash changes, unknown ids |
| `check_mobile` | no horizontal scroll, touch targets ≥ 44px, swipes can scroll over the plot, taps work (WebKit uses the iPhone 13 profile) |
| `make_screenshots` | `notes/screenshots/*.png`, `scrub.gif`, `site/og-image.png` |

## Deployment — `.github/workflows/pages.yml`

On every push to `main` (or a manual run), the workflow:
1. checks out the repo
2. uploads `site/` as the Pages artifact
3. deploys it with `actions/deploy-pages`

It has the permissions Pages needs (`pages: write`, `id-token: write`,
`actions: read`), and a `concurrency` group so deploys never overlap. There is
no build step: the site is the files in `site/`.

## Adding a run

1. Add a `RunSpec` to `SWEEP` in `train/config.py`, changing one field of
   `BASELINE`.
2. `python train/train.py --run <id>`, then
   `python train/project.py <all ids>`, `python train/export.py`, and
   `python train/validate.py`.
3. `validate.py` and `check_site.py` hard-code the four current run ids
   (`EXPECTED_ORDER`, `check_toggle`), so update those lists too. The page
   itself builds its buttons from the manifest and needs no changes.
