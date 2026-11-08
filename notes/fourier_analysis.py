"""Fourier analysis of the number-token embeddings (standalone).

Reads <outputs>/<run>/embeddings.npy (snapshots x p x d_model) and run.json for
each run, plus the projected coordinates the site renders, and writes to
<notes>/:

    fourier_spectrum.png         final spectrum, one panel per run
    fourier_over_training.png    key-frequency energy vs test acc, one panel per run
    fourier_m3_summary.csv       one row per seed: steps, key freqs, timing, subspaces
    fourier_m3_spectrum.csv      energy fraction of every frequency, per seed
    fourier_m3_over_training.csv one row per embedding snapshot, per seed
    fourier_m3_pcs.csv           PC-vs-frequency comparison, per seed
    fourier_site_axes.csv        which frequencies the site's plotted axes contain

    python notes/fourier_analysis.py
    python notes/fourier_analysis.py --outputs outputs \
        --runs wd1.0__seed0 wd1.0__seed1 wd1.0__seed2 \
        --site-run site/data/runs/baseline.json --notes notes

Conventions
- Real Fourier basis over the token axis x = 0..p-1: the constant vector, and
  for k = 1..(p-1)/2 the unit-norm vectors cos(2*pi*k*x/p), sin(2*pi*k*x/p).
  These p vectors form an orthonormal basis, so the squared Frobenius norm of
  W_E splits exactly into per-frequency parts.
- "Fraction" always means fraction of the squared norm (energy) of the
  mean-centred embedding (the k=0 constant component is excluded), so it is
  the same quantity PCA explains variance of.
- "Grokking step" and "50% crossing" are the first *recorded* step with test
  accuracy >= 95% / >= 50% (the repo's definition, `first_step` in train/plot.py).
- Embedding snapshots are sparser than metric records. When a step has no
  snapshot, the energy fraction there is linearly interpolated in step between
  the two neighbouring snapshots; both neighbours are written to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOP = 5  # number of frequencies called "key" (Nanda et al. report ~5 for p=113)
N_PCS = 8

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#b8b6ae"
INK = "#52514e"
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e8e6df",
    "grid.linewidth": 0.8,
})


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
    e = ((C @ Wc) ** 2).sum(1) + ((S @ Wc) ** 2).sum(1)
    total = (Wc**2).sum()
    assert np.isclose(e.sum(), total, rtol=1e-4), (e.sum(), total)  # basis is complete
    return ks, e, total


def first_step(run: dict, key: str, threshold: float):
    for s, a in zip(run["steps"], run[key]):
        if a >= threshold:
            return s
    return None


def value_at(steps: np.ndarray, values: np.ndarray, step: int):
    """(step_before, value_before, step_after, value_after, interpolated)."""
    j = int(np.searchsorted(steps, step))
    if j < len(steps) and steps[j] == step:
        return int(step), float(values[j]), int(step), float(values[j]), float(values[j])
    a, b = j - 1, j
    t = (step - steps[a]) / (steps[b] - steps[a])
    return (int(steps[a]), float(values[a]), int(steps[b]), float(values[b]),
            float(values[a] + t * (values[b] - values[a])))


def freq_r2(v: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Share of the zero-mean vector v in each frequency (any phase); sums to 1."""
    u = (v - v.mean()) / np.linalg.norm(v - v.mean())
    return (C @ u) ** 2 + (S @ u) ** 2


def subspace_cosines(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Principal-angle cosines between two column spans (basis independent)."""
    return np.linalg.svd(np.linalg.qr(P)[0].T @ np.linalg.qr(Q)[0], compute_uv=False)


def freq_columns(C, S, ks, chosen) -> np.ndarray:
    return np.concatenate([np.stack([C[ks == k][0], S[ks == k][0]], 1) for k in chosen], 1)


def fit_pca_site(W: np.ndarray):
    """Same fit as train/project.py: 2 PCs of the final snapshot, sign-fixed."""
    mean = W.mean(0)
    _, _, Vt = np.linalg.svd(W - mean, full_matrices=False)
    comps = Vt[:2]
    signs = np.sign(comps[np.arange(2), np.abs(comps).argmax(axis=1)])
    return mean, comps * signs[:, None]


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name} ({len(rows)} rows)")


def f4(x) -> str:
    return f"{x:.4f}"


def ids(values) -> str:
    return ";".join(str(int(v)) for v in values)


def analyse_axes(source: str, coords: np.ndarray, ks, C, S) -> tuple[list[dict], dict]:
    """Frequency content of a p x 2 array of plotted coordinates."""
    rows, shares = [], {}
    x, y = coords[:, 0] - coords[:, 0].mean(), coords[:, 1] - coords[:, 1].mean()
    for name, v in (("x(PC1)", x), ("y(PC2)", y)):
        r2 = freq_r2(v, C, S)
        shares[name] = r2
        for rank, i in enumerate(np.argsort(r2)[::-1][:3], 1):
            rows.append(dict(source=source, quantity="axis_r2", axis=name, rank=rank,
                             k=int(ks[i]), value=f4(r2[i])))
    vx, vy = (x**2).sum(), (y**2).sum()
    plane = (shares["x(PC1)"] * vx + shares["y(PC2)"] * vy) / (vx + vy)
    for rank, i in enumerate(np.argsort(plane)[::-1][:4], 1):
        rows.append(dict(source=source, quantity="plane_variance_share", axis="plane",
                         rank=rank, k=int(ks[i]), value=f4(plane[i])))
    P = np.stack([x, y], 1)
    cosines = [subspace_cosines(P, freq_columns(C, S, ks, [k])) for k in ks]
    best = int(np.argmax([c.min() for c in cosines]))
    for rank, cos in enumerate(sorted(cosines[best], reverse=True), 1):
        rows.append(dict(source=source, quantity="best_single_k_principal_cosine", axis="plane",
                         rank=rank, k=int(ks[best]), value=f4(cos)))
    return rows, shares


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs",
                    help="directory holding one sub-directory per run")
    ap.add_argument("--runs", nargs="+",
                    default=["wd1.0__seed0", "wd1.0__seed1", "wd1.0__seed2"],
                    help="run directories under --outputs, in the order to plot")
    ap.add_argument("--site-run", type=Path, default=ROOT / "site/data/runs/baseline.json",
                    help="site JSON whose final frame is analysed directly")
    ap.add_argument("--notes", type=Path, default=ROOT / "notes", help="where outputs are written")
    args = ap.parse_args()

    data = {}
    for name in args.runs:
        d = args.outputs / name
        run = json.loads((d / "run.json").read_text())
        data[int(run["config"]["seed"])] = (name, np.load(d / "embeddings.npy"), run)
    p = next(iter(data.values()))[2]["config"]["p"]
    ks, C, S = fourier_basis(p)
    uniform = TOP / len(ks)
    print(f"p = {p}; uniform share of {TOP} frequencies = {uniform:.4f}")

    summary, spectrum_rows, training_rows, pc_rows = [], [], [], []
    keys, timing, curves = {}, {}, {}

    fig, axes = plt.subplots(len(data), 1, figsize=(9, 2.6 * len(data)), sharex=True, squeeze=False)
    for ax, (seed, (name, E, run)) in zip(axes[:, 0], sorted(data.items())):
        # ---- final spectrum ---------------------------------------------------
        _, e, total = spectrum(E[-1])
        frac = e / total
        order = np.argsort(frac)[::-1]
        key = ks[order[:TOP]]
        keys[seed] = key
        cum = np.cumsum(frac[order])
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        spectrum_rows += [dict(seed=seed, k=int(k), energy_fraction=f"{f:.6f}") for k, f in zip(ks, frac)]

        ax.bar(ks, frac, width=0.7, color=GREY)
        ax.bar(key, frac[order[:TOP]], width=0.7, color=BLUE)
        for k, f in zip(key, frac[order[:TOP]]):
            ax.annotate(str(int(k)), (k, f), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_ylabel("energy fraction")
        ax.set_ylim(0, 0.36)
        ax.set_title(f"Seed {seed}: top 5 frequencies carry {cum[TOP - 1]:.1%} of centred energy",
                     fontsize=10, loc="left", color=INK)

        # ---- PCs vs frequencies -----------------------------------------------
        Wc = E[-1] - E[-1].mean(0)
        U, Sv, _ = np.linalg.svd(Wc, full_matrices=False)
        evr = Sv**2 / (Sv**2).sum()
        for j in range(N_PCS):
            r2 = freq_r2(U[:, j], C, S)
            top = np.argsort(r2)[::-1][:2]
            pc_rows.append(dict(seed=seed, pc=j + 1, variance_fraction=f4(evr[j]),
                                best_k=int(ks[top[0]]), best_r2=f4(r2[top[0]]),
                                second_k=int(ks[top[1]]), second_r2=f4(r2[top[1]])))

        def overlap(m):
            sv = subspace_cosines(U[:, :2 * m], freq_columns(C, S, ks, key[:m]))
            return float(sv.min()), float(np.mean(sv**2))

        min8, mean8 = overlap(4)

        # ---- over training ----------------------------------------------------
        snap = np.array(run["embedding_steps"])
        key_frac, top_frac = [], []
        for W in E:
            k_all, en, tot = spectrum(W)
            key_frac.append(en[np.isin(k_all, key)].sum() / tot)
            top_frac.append(np.sort(en)[::-1][:TOP].sum() / tot)
        key_frac, top_frac = np.array(key_frac), np.array(top_frac)
        rec = np.array(run["steps"])
        test, train = np.array(run["test_acc"]), np.array(run["train_acc"])
        for s, kf, tf in zip(snap, key_frac, top_frac):
            i = int(np.searchsorted(rec, s))
            training_rows.append(dict(seed=seed, step=int(s), test_acc=f4(test[i]), train_acc=f4(train[i]),
                                      key_freq_energy=f4(kf), top5_at_step_energy=f4(tf)))

        s50, s95 = first_step(run, "test_acc", 0.5), first_step(run, "test_acc", 0.95)
        s99tr = first_step(run, "train_acc", 0.99)
        a50, a95 = test[np.searchsorted(rec, s50)], test[np.searchsorted(rec, s95)]
        at50 = value_at(snap, key_frac, s50)
        at95 = value_at(snap, key_frac, s95)
        at99 = value_at(snap, key_frac, s99tr)
        per_step = np.diff(key_frac) / np.diff(snap)
        per_log = np.diff(key_frac) / np.diff(np.log(snap))
        fast, fast_log = int(np.argmax(per_step)), int(np.argmax(per_log))
        snap_test = test[np.searchsorted(rec, snap)]
        pre = int(np.argmax(snap_test >= 0.1)) - 1  # last snapshot before test acc reaches 10%
        summary.append(dict(
            seed=seed, run=name, final_step=int(snap[-1]), final_test_acc=f4(test[-1]),
            grokking_step=s95, test_acc_at_grokking_step=f4(a95),
            test50_step=s50, test_acc_at_test50_step=f4(a50), train99_step=s99tr,
            key_freqs=ids(key), key_fractions=";".join(f"{x:.3f}" for x in frac[order[:TOP]]),
            top4_energy=f4(cum[3]), top5_energy=f4(cum[4]), freqs_for_90pct=n90,
            max_outside_top5=f4(frac[order[TOP]]),
            first_snapshot_step=int(snap[0]), key_energy_first_snapshot=f4(key_frac[0]),
            key_energy_at_train99=f4(at99[4]),
            before_test10_snapshot=int(snap[pre]), before_test10_test_acc=f4(snap_test[pre]),
            before_test10_key_energy=f4(key_frac[pre]),
            key_energy_at_test50=f4(at50[4]),
            grok_snapshot_before=at95[0], key_energy_before=f4(at95[1]),
            grok_snapshot_after=at95[2], key_energy_after=f4(at95[3]),
            key_energy_at_grokking=f4(at95[4]),
            key_energy_final=f4(key_frac[-1]),
            fastest_per_step_from=int(snap[fast]), fastest_per_step_to=int(snap[fast + 1]),
            fastest_per_log_step_from=int(snap[fast_log]), fastest_per_log_step_to=int(snap[fast_log + 1]),
            fastest_per_log_step_test_acc_from=f4(snap_test[fast_log]),
            fastest_per_log_step_test_acc_to=f4(snap_test[fast_log + 1]),
            top8pc_vs_top4freq_min_cos=f4(min8), top8pc_vs_top4freq_mean_cos2=f4(mean8),
        ))
        timing[seed] = (s50, s95)
        curves[seed] = (snap, key_frac, rec, test)
        print(f"[seed {seed}] grok {s95}, 50% at {s50}, key {sorted(key.tolist())}, "
              f"energy at 50% {at50[4]:.3f}, at grok {at95[4]:.3f}")

    axes[-1, 0].set_xlabel("frequency k (each k is a cos/sin pair)")
    fig.suptitle(f"Fourier spectrum of final number-token embeddings (p={p}, M3 runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.notes / "fourier_spectrum.png", dpi=130)
    plt.close(fig)

    # ---- the site's plotted axes, read from the data the page renders --------
    site = json.loads(args.site_run.read_text())
    frame = site["embedding_frames"][-1]
    site_xy = np.array(frame["coords"])
    site_rows, _ = analyse_axes(f"site:{args.site_run.name}", site_xy, ks, C, S)
    seed0 = data.get(int(site["config"]["seed"]))
    if seed0 is not None:
        name, E, _ = seed0
        mean, comps = fit_pca_site(E[-1])
        xy = (E[-1] - mean) @ comps.T
        rows, _ = analyse_axes(f"recomputed:{name}", xy, ks, C, S)
        site_rows += rows
        site_rows.append(dict(source=f"site vs recomputed:{name}", quantity="max_abs_coord_diff",
                              axis="both", rank=1, k="", value=f"{np.abs(xy - site_xy).max():.6f}"))
    site_rows.append(dict(source=f"site:{args.site_run.name}", quantity="frame_step", axis="", rank=1,
                          k="", value=frame["step"]))

    # ---- energy vs test accuracy, one panel per seed --------------------------
    fig, axes = plt.subplots(len(curves), 1, figsize=(9, 2.8 * len(curves)), sharex=True, squeeze=False)
    handles = None
    for ax, (seed, (snap, key_frac, rec, test)) in zip(axes[:, 0], sorted(curves.items())):
        s50, s95 = timing[seed]
        l1, = ax.plot(snap, key_frac, "-o", ms=3, lw=2, color=BLUE, label="energy in the seed's final key frequencies")
        l2, = ax.plot(rec, test, "-", lw=2, color=ORANGE, label="test accuracy")
        l3 = ax.axhline(uniform, color=INK, lw=1, ls=":", label=f"even spread over 48 frequencies ({uniform:.3f})")
        l4 = ax.axvline(s50, color=INK, lw=1, ls="--", label="test accuracy first ≥ 50%")
        l5 = ax.axvline(s95, color=INK, lw=1.2, ls="-", label="grokking step (test accuracy first ≥ 95%)")
        handles = [l1, l2, l3, l4, l5]
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("fraction")
        ax.set_title(f"Seed {seed}: key frequencies {sorted(keys[seed].tolist())}; "
                     f"50% at step {s50:,}, grokking at step {s95:,}",
                     fontsize=10, loc="left", color=INK)
    axes[-1, 0].set_xlabel("training step (log scale)")
    fig.suptitle("Energy in final key frequencies vs test accuracy (M3 runs)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(args.notes / "fourier_over_training.png", dpi=130)
    plt.close(fig)

    write_csv(args.notes / "fourier_m3_summary.csv", summary)
    write_csv(args.notes / "fourier_m3_spectrum.csv", spectrum_rows)
    write_csv(args.notes / "fourier_m3_over_training.csv", training_rows)
    write_csv(args.notes / "fourier_m3_pcs.csv", pc_rows)
    write_csv(args.notes / "fourier_site_axes.csv", site_rows)


if __name__ == "__main__":
    main()
