# Are the baseline embedding rings Fourier features?

Question: the site shows the number-token embeddings of the `baseline` run
organising into rings. Is that structure the sparse Fourier representation
reported by Nanda et al. 2023 ("Progress measures for grokking via mechanistic
interpretability")? This note tests that **at the embedding level only**.

Script: `notes/fourier_analysis.py` (standalone; reads `outputs/`).
Reproduce:

```bash
python train/train.py --run baseline              # and --seed 1, --seed 2
python notes/fourier_analysis.py
```

## Setup

- Task `(a + b) mod p`, **p = 97**, 30% train split, weight decay 1.0, 25k
  full-batch AdamW steps. Model: 1 layer, 4 heads, d_model 128, d_mlp 512, no
  LayerNorm, logits read from the last (`=`) position (`train/config.py`,
  `train/model.py`).
- The runs were retrained in this session on CPU with the repo's own command.
  `embeddings.npy` has shape **(51, 97, 128)**: 51 snapshots of the number-token
  rows of `W_E` over training.
- Steps where test accuracy first reached 95% in these runs: seed 0 at 10,065,
  seed 1 at 5,452 and seed 2 at 3,973. The README reports 4,216, 5,785 and 9,868
  for its own runs, which is the same set of grokking times but in a different
  seed order. That was not investigated: the possible causes are a different
  torch version or thread count. All three runs end at 100% test accuracy.

## Method

- **Basis.** Over the token axis x = 0..96, use the orthonormal real Fourier
  basis: one constant vector plus unit-norm cos(2πkx/p) and sin(2πkx/p) vectors
  for k = 1..48. The rows of `W_E` (97 × 128) are mean-centred, which removes
  the k = 0 component, and then projected onto this basis.
- **Energy of a frequency.** The energy of frequency k is |cᵀW|² + |sᵀW|². The
  basis is complete, so the per-k energies add up exactly to ‖W − mean‖²_F, and
  the script asserts this. "Fraction" below always means a share of this energy,
  which is the same quantity whose variance PCA explains.
- **PCs.** These come from an SVD of the centred final snapshot, the same
  centring and decomposition as `train/project.py`, so PC1 and PC2 are the
  site's axes. Each PC is compared with cos/sin of every k. The comparison is
  also made between subspaces using principal angles, which does not depend on
  how the basis inside a subspace is rotated.
- **Key frequencies.** These are the top 5 frequencies by energy in the final
  snapshot.

## Results

### 1. Spectrum of final W_E

![spectrum](fourier_spectrum.png)

| seed | top frequencies (energy fraction) | top-4 | top-5 | freqs for 90% |
|---|---|---|---|---|
| 0 | 12 (.265), 3 (.260), 32 (.250), 10 (.188), 20 (.026) | 0.963 | 0.990 | 4 / 48 |
| 1 | 33 (.290), 14 (.243), 34 (.231), 21 (.220), 31 (.003) | 0.984 | 0.988 | 4 / 48 |
| 2 | 34 (.291), 9 (.243), 1 (.223), 41 (.177), 15 (.059) | 0.934 | 0.993 | 4 / 48 |

If energy were spread evenly, 5 of the 48 frequencies would carry 0.104. In
every seed the final number embeddings are close to fully Fourier-sparse:
**4 frequencies carry 93–99% of the centred energy**, and every other frequency
carries no more than 0.6%, apart from a weak fifth component (k = 20 at 2.6% in
seed 0, k = 15 at 5.9% in seed 2).

### 2. PCs ↔ frequencies (seed 0)

| PC | var | best k | R² onto (cos k, sin k) |
|---|---|---|---|
| 1 | 13.5% | 12 | 0.960 |
| 2 | 13.3% | 3 | 0.890 |
| 3 | 13.1% | 12 | 0.882 |
| 4 | 12.8% | 3 | 0.977 |
| 5 | 12.6% | 32 | 0.907 |
| 6 | 12.4% | 32 | 0.995 |

- The top 8 PCs span essentially the same 8-D space as cos/sin of
  {3, 10, 12, 32}: the smallest principal-angle cosine is 0.998.
- **The site's 2D view is not a single cos/sin pair.** The top PC variances are
  nearly equal (13.5, 13.3, 13.1, 12.8, …), so the PCA basis within this
  subspace is close to arbitrary. PC1 is 96% frequency 12 and PC2 is 89%
  frequency 3. The best single frequency (k = 12) matches the PC1–PC2 plane in
  only one direction: its principal-angle cosines are 0.998 and 0.248.
- So the ring-like shapes in the site's scatter are a 2D projection that mixes
  two Fourier frequencies, a Lissajous-type curve, and not the clean
  cos(2πkx/p), sin(2πkx/p) circle of one frequency. The underlying representation
  is Fourier, but the specific ring the site shows is a mix of frequencies.

### 3. Over training (seed 0)

![over training](fourier_over_training.png)

The red curve is the energy fraction in seed 0's final key frequencies
{3, 10, 12, 20, 32}. The orange curve is the energy in whichever 5 frequencies
are largest at each step, which is basis-free.

| step | test acc | key-freq fraction |
|---|---|---|
| 1 | 1.1% | 0.109 (≈ uniform 0.104) |
| 125 (train ≈ 99%) | 1.5% | 0.13 |
| 3,599 | 7.3% | 0.20 |
| 5,785 | 11.5% | 0.25 |
| 8,939 | 54.7% | 0.42 |
| 9,868 | 93.9% | 0.55 |
| 10,471 | 99.3% | 0.63 |
| 12,266 | 100% | 0.76 |
| 15,552 | 100% | 0.91 |
| 25,000 | 100% | 0.99 |

**Timing.** Sparsity begins to rise **before** the test-accuracy jump. It climbs
slowly from roughly step 100, when the training set is memorised, reaching 0.20
by step 3.6k while test accuracy is still about 7%. It rises **fastest during**
the jump, from 0.25 to 0.63 between steps 5.8k and 10.5k. It is **far from
finished** when test accuracy reaches 99%, at only 0.63, and keeps rising to
0.99 by step 25k.

This fits Nanda et al.'s account of a gradual "circuit formation" phase followed
by "cleanup" after generalisation. It is a single seed with snapshots only, so
the timing is measured for seed 0 alone.

### 4. Seeds

The key-frequency sets have **no frequency shared by all three seeds**:

- seed 0: {3, 10, 12, (20), 32}
- seed 1: {14, 21, 33, 34}
- seed 2: {1, 9, 34, 41, (15)}

Only k = 34 appears in two seeds (1 and 2). This matches Nanda et al.'s
observation that which frequencies are used depends on the seed, while the
sparsity itself does not.

## Confirmed / Not tested

**Confirmed (embedding level, this repo, p = 97, 3 seeds):**

- Final number-token embeddings are Fourier-sparse. 4 of 48 frequencies carry
  93–99% of the centred energy.
- The top PCs line up with cos/sin of those key frequencies. The top 8 PCs and
  the 4 key cos/sin pairs span the same space, with principal-angle cosine
  ≥ 0.998 (seed 0).
- The site's PC1/PC2 axes are each dominated by one key frequency. The two axes
  belong to *different* frequencies (12 and 3 for seed 0), so the plotted ring
  is a mix of frequencies.
- Fourier sparsity rises slowly before the test-accuracy jump, fastest during
  it, and keeps rising long after it (seed 0 only).
- Key frequencies differ across seeds.

**Not tested:**

- Whether the model *uses* these frequencies to compute the answer. Not checked:
  the logit-level trig-identity form (logits ∝ Σ_k cos(2πk(a+b−c)/p)), `W_U`,
  the attention or MLP neurons, and whether neuron activations are products of
  cos/sin terms in a and b.
- Any ablation, such as removing the key frequencies from `W_E` or the logits
  and checking whether accuracy collapses, or keeping only them and checking
  whether accuracy holds.
- Nanda's progress measures (restricted/excluded loss).
- The other runs (`no_decay`, `more_data`, `less_data`), and Fourier timing for
  seeds 1 and 2.

So this note does **not** show that the model implements the Fourier
multiplication algorithm. It shows that its embeddings carry the Fourier
representation that algorithm would read from.
