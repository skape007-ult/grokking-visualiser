"""Fourier analysis of the number-token embeddings (standalone).

Reads outputs/<run>/embeddings.npy (snapshots x p x d_model) and run.json,
writes notes/fourier_spectrum.png, notes/fourier_over_training.png and prints
the numbers quoted in notes/fourier.md.

    python notes/fourier_analysis.py            # baseline seeds 0, 1, 2 (whichever exist)

Conventions
- Real Fourier basis over the token axis x = 0..p-1: the constant vector, and
  for k = 1..(p-1)/2 the unit-norm vectors cos(2*pi*k*x/p), sin(2*pi*k*x/p).
  These p vectors form an orthonormal basis, so the squared Frobenius norm of
  W_E splits exactly into per-frequency parts.
- "Norm of frequency k" = sqrt(|c_k^T W_E|^2 + |s_k^T W_E|^2).
- "Fraction" always means fraction of the squared norm (energy), of the
  mean-centred embedding (the k=0 constant component is excluded), so it is
  the same quantity PCA explains variance of.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
NOTES = ROOT / "notes"
RUNS = {0: "baseline", 1: "baseline__seed1", 2: "baseline__seed2"}
TOP = 5  # number of frequencies called "key" (Nanda et al. report ~5 for p=113)


def fourier_basis(p: int):
    x = np.arange(p)
    ks = np.arange(1, (p - 1) // 2 + 1)
    C = np.cos(2 * np.pi * np.outer(ks, x) / p)
    S = np.sin(2 * np.pi * np.outer(ks, x) / p)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    return ks, C, S


def spectrum(W: np.ndarray):
    """Per-frequency energy (k=1..(p-1)/2) of the mean-centred W (p x d)."""
    ks, C, S = fourier_basis(W.shape[0])
    Wc = W - W.mean(0)
    e = (C @ Wc) ** 2
    e = e.sum(1) + ((S @ Wc) ** 2).sum(1)
    total = (Wc**2).sum()
    assert np.isclose(e.sum(), total, rtol=1e-4), (e.sum(), total)  # basis is complete
    return ks, e, total


def load(seed: int):
    d = OUT / RUNS[seed]
    if not (d / "embeddings.npy").exists():
        return None
    return np.load(d / "embeddings.npy"), json.loads((d / "run.json").read_text())


def step_at(run, thr):
    for s, a in zip(run["steps"], run["test_acc"]):
        if a >= thr:
            return s
    return None


def main():
    data = {s: load(s) for s in RUNS}
    data = {s: v for s, v in data.items() if v is not None}
    assert 0 in data, "train baseline seed 0 first: python train/train.py --run baseline"
    emb, run = data[0]
    p = run["config"]["p"]
    print(f"p = {p}, embeddings.npy shape = {emb.shape} (snapshots x p x d_model)")

    # ---- 1. spectrum of the final W_E, per seed -------------------------------
    key = {}
    fig, axes = plt.subplots(len(data), 1, figsize=(9, 2.4 * len(data)), sharex=True, squeeze=False)
    for ax, (seed, (E, r)) in zip(axes[:, 0], sorted(data.items())):
        ks, e, total = spectrum(E[-1])
        order = np.argsort(e)[::-1]
        key[seed] = ks[order[:TOP]]
        frac = e / total
        cum = np.cumsum(frac[order])
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        print(f"\n[seed {seed}] final step {r['embedding_steps'][-1]}")
        print("  top frequencies (k: energy fraction):",
              ", ".join(f"{ks[i]}: {frac[i]:.3f}" for i in order[:10]))
        print(f"  top-{TOP} carry {cum[TOP - 1]:.3f} of energy; "
              f"{n90} of {len(ks)} frequencies needed for 90%")
        print(f"  uniform baseline for {TOP} freqs: {TOP / len(ks):.3f}")
        norms = np.sqrt(e)
        ax.bar(ks, norms, color="#888")
        ax.bar(ks[order[:TOP]], norms[order[:TOP]], color="#c0392b")
        for i in order[:TOP]:
            ax.text(ks[i], norms[i], str(ks[i]), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("norm")
        ax.set_title(f"seed {seed}: top-{TOP} = {sorted(key[seed].tolist())} "
                     f"carry {cum[TOP - 1]:.0%} of centred W_E energy", fontsize=9)
    axes[-1, 0].set_xlabel("frequency k (cos/sin pair)")
    fig.suptitle(f"Fourier spectrum of final number-token embeddings (p={p})", fontsize=10)
    fig.tight_layout()
    fig.savefig(NOTES / "fourier_spectrum.png", dpi=130)
    if len(data) > 1:
        sets = {s: set(v.tolist()) for s, v in key.items()}
        print("\nkey-frequency sets per seed:", {s: sorted(v) for s, v in sets.items()})
        print("  shared by all seeds:", sorted(set.intersection(*sets.values())))

    # ---- 2. PCs vs cos/sin of each frequency (seed 0, final snapshot) ---------
    W = emb[-1]
    Wc = W - W.mean(0)
    U, Sv, Vt = np.linalg.svd(Wc, full_matrices=False)
    evr = Sv**2 / (Sv**2).sum()
    scores = U * Sv  # p x n_pc, PC coordinate of each token
    ks, C, S = fourier_basis(p)
    print("\n[seed 0] PCs of final W_E (same centring/SVD as train/project.py)")
    for j in range(6):
        pc = scores[:, j] / np.linalg.norm(scores[:, j])
        rc, rs = C @ pc, S @ pc  # cosine similarity (both zero-mean, unit norm)
        r2 = rc**2 + rs**2       # share of this PC explained by frequency k (any phase)
        b = int(np.argmax(r2))
        print(f"  PC{j + 1} ({evr[j]:.1%} var): best k={ks[b]}  corr cos={rc[b]:+.3f} "
              f"sin={rs[b]:+.3f}  R^2(cos,sin)={r2[b]:.3f}")
    # Phase-invariant check for the 2D ring the site shows: does the PC1/PC2
    # plane coincide with the (cos k, sin k) plane? Principal-angle cosines.
    for j0 in (0, 2):
        P = scores[:, j0:j0 + 2] / np.linalg.norm(scores[:, j0:j0 + 2], axis=0)
        best = None
        for i, k in enumerate(ks):
            Q = np.stack([C[i], S[i]], 1)
            sv = np.linalg.svd(P.T @ Q, compute_uv=False)
            if best is None or sv.min() > best[1].min():
                best = (k, sv)
        print(f"  plane PC{j0 + 1}-PC{j0 + 2} vs (cos,sin) of k={best[0]}: "
              f"principal-angle cosines {best[1].round(3).tolist()}")

    # ---- 3. over training (seed 0) --------------------------------------------
    steps = np.array(run["embedding_steps"])
    fixed = key[0]
    f_fixed, f_top = [], []
    for E in emb:
        ks_, e, total = spectrum(E)
        f_fixed.append(e[np.isin(ks_, fixed)].sum() / total)
        f_top.append(np.sort(e)[::-1][:TOP].sum() / total)
    f_fixed, f_top = np.array(f_fixed), np.array(f_top)
    ms, ta, tra = np.array(run["steps"]), np.array(run["test_acc"]), np.array(run["train_acc"])
    print("\n[seed 0] over training: step, test_acc, frac(final key freqs), frac(top-5 at that step)")
    for s, a, b in zip(steps, f_fixed, f_top):
        acc = ta[np.searchsorted(ms, s)]
        print(f"  {s:>6}  test {acc:6.1%}  key {a:.3f}  top5 {b:.3f}")
    for thr in (0.1, 0.5, 0.95):
        print(f"  test acc >= {thr:.0%} at step {step_at(run, thr)}")
    uni = TOP / len(ks)
    for lvl in (0.2, 0.3, 0.5, 0.7):
        hit = steps[np.argmax(f_fixed >= lvl)] if (f_fixed >= lvl).any() else None
        print(f"  key-freq fraction >= {lvl} first at snapshot step {hit}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, f_fixed, "o-", ms=3, color="#c0392b", label=f"energy in final key freqs {sorted(fixed.tolist())}")
    ax.plot(steps, f_top, "s--", ms=2.5, color="#e67e22", label=f"energy in top-{TOP} freqs at that step")
    ax.axhline(uni, color="#c0392b", lw=0.6, ls=":", label=f"uniform spread ({uni:.2f})")
    ax.plot(ms, ta, color="#2c3e50", lw=1.5, label="test accuracy")
    ax.plot(ms, tra, color="#95a5a6", lw=1, label="train accuracy")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("fraction")
    ax.set_title("Seed 0: Fourier sparsity of W_E vs accuracy", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(NOTES / "fourier_over_training.png", dpi=130)


if __name__ == "__main__":
    main()
