"""less_data at 100,000 steps on the M3: notes/less_data_long.{csv,png} and the tables in the note.

    python notes/less_data_long.py

Reads outputs/long100k__seed<S>/run.json for seeds 0, 1 and 2 (a missing or incomplete run is an
error), produced by

    python train/train.py --run less_data --seed S --set steps=100000 --tag long100k__seedS

and notes/less_data_long_cloud60k.csv, the 60k cloud-CPU curves from PR #2, copied verbatim.

Memorisation and grokking use the repo's definition: the first *recorded* step with train acc >= 99%
and test acc >= 95% (train/plot.py: first_step).

Test-loss trend, fixed before any 100k result was seen: ordinary least-squares line of test loss
against step over a window. The trend is "flat" if the fitted change across the window is under
FLAT_FRAC of the window's mean test loss, otherwise "increasing" or "falling" by its sign.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from config import RUNS  # noqa: E402
from plot import first_step  # noqa: E402  (the repo's definition)

SEEDS = [0, 1, 2]
STEPS = 100_000
CLOUD_STEPS = 60_000
TRAIN_THRESH = 0.99
TEST_THRESH = 0.95
UNIFORM_LOSS = math.log(97)
CHANCE = 1 / 97
FLAT_FRAC = 0.01
WINDOWS = [(50_000, 100_000), (80_000, 100_000)]
CHECKPOINTS = [1_000, 10_000, 25_000, 50_000, 75_000, 100_000]
COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}  # categorical slots 1-3, in order


def load_m3(seed):
    f = ROOT / "outputs" / f"long100k__seed{seed}" / "run.json"
    d = json.loads(f.read_text())  # a missing run is an error
    expected = RUNS["less_data"].config.with_overrides(seed=seed, steps=STEPS).to_dict()
    assert d["config"] == expected, f"{f}: config differs from less_data + seed/steps"
    assert d["runtime"]["device"] == "cpu", f
    assert d["steps"][-1] == STEPS, f"{f}: stopped at step {d['steps'][-1]}"
    return d


def load_cloud():
    runs = {s: {"steps": [], "train_acc": [], "test_acc": []} for s in SEEDS}
    with open(ROOT / "notes" / "less_data_long_cloud60k.csv", newline="") as f:
        for r in csv.DictReader(f):
            run = runs[int(r["seed"])]
            run["steps"].append(int(r["step"]))
            run["train_acc"].append(float(r["train_acc"]))
            run["test_acc"].append(float(r["test_acc"]))
    for s, run in runs.items():
        assert run["steps"][-1] == CLOUD_STEPS, f"cloud seed {s}: stopped at step {run['steps'][-1]}"
    return runs


def at_or_before(d, key, step):
    """(recorded step, value) at the last recorded step <= step."""
    i = int(np.searchsorted(d["steps"], step, side="right")) - 1
    return d["steps"][i], d[key][i]


def window_trend(d, lo, hi):
    pts = [(s, l) for s, l in zip(d["steps"], d["test_loss"]) if lo <= s <= hi]
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    change = slope * (hi - lo)
    mean = float(y.mean())
    label = "flat" if abs(change) < FLAT_FRAC * mean else ("increasing" if change > 0 else "falling")
    return {"n": len(pts), "first": float(y[0]), "last": float(y[-1]), "change": change, "mean": mean,
            "pct": 100 * change / mean, "label": label}


def summary(seed, d):
    i_peak = int(np.argmax(d["test_loss"]))
    memorised = first_step(d["steps"], d["train_acc"], TRAIN_THRESH)
    late_test = [a for s, a in zip(d["steps"], d["test_acc"]) if s >= 50_000]
    return {
        "seed": seed,
        "max_test_after_mem": max(a for s, a in zip(d["steps"], d["test_acc"]) if s >= memorised),
        "late_test_min": min(late_test),
        "late_test_max": max(late_test),
        "memorised": memorised,
        "grok": first_step(d["steps"], d["test_acc"], TEST_THRESH),
        "final_train": d["train_acc"][-1],
        "final_test": d["test_acc"][-1],
        "max_test": max(d["test_acc"]),
        "max_test_step": d["steps"][int(np.argmax(d["test_acc"]))],
        "final_test_loss": d["test_loss"][-1],
        "peak_test_loss": d["test_loss"][i_peak],
        "peak_test_loss_step": d["steps"][i_peak],
        "seconds": d["runtime"]["seconds"],
        "n_recorded": len(d["steps"]),
    }


def compare(seed, m3, cloud):
    """M3 vs cloud over the first 60k steps: M3 accuracy interpolated (in log step) at the cloud's steps."""
    cs = np.array(cloud["steps"], dtype=float)
    out = {"seed": seed, "points": len(cs)}
    for key in ("train_acc", "test_acc"):
        m = np.interp(np.log(cs), np.log(np.array(m3["steps"], dtype=float)), m3[key])
        delta = np.abs(m - np.array(cloud[key]))
        out[f"{key}_median"] = float(np.median(delta))
        out[f"{key}_max"] = float(delta.max())
        late = cs >= 1000
        out[f"{key}_late_max"] = float(delta[late].max())
    upto = [i for i, s in enumerate(m3["steps"]) if s <= CLOUD_STEPS]
    out["m3_memorised"] = first_step(m3["steps"], m3["train_acc"], TRAIN_THRESH)
    out["cloud_memorised_csv"] = first_step(cloud["steps"], cloud["train_acc"], TRAIN_THRESH)
    out["m3_max_test_60k"] = max(m3["test_acc"][i] for i in upto)
    out["cloud_max_test_csv"] = max(cloud["test_acc"])
    out["m3_step_near_60k"], out["m3_test_near_60k"] = at_or_before(m3, "test_acc", CLOUD_STEPS)
    out["m3_test_loss_near_60k"] = at_or_before(m3, "test_loss", CLOUD_STEPS)[1]
    out["cloud_test_60k"] = cloud["test_acc"][-1]
    return out


def write_csv(m3, cloud, path):
    fields = ["source", "seed", "step", "train_acc", "test_acc", "train_loss", "test_loss"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for s in SEEDS:
            d = m3[s]
            for i, step in enumerate(d["steps"]):
                w.writerow(["m3_100k", s, step, f"{d['train_acc'][i]:.6f}", f"{d['test_acc'][i]:.6f}",
                            f"{d['train_loss'][i]:.6g}", f"{d['test_loss'][i]:.6g}"])
        for s in SEEDS:
            c = cloud[s]
            for i, step in enumerate(c["steps"]):
                w.writerow(["cloud_60k", s, step, f"{c['train_acc'][i]:.4f}", f"{c['test_acc'][i]:.4f}", "", ""])


def plot(m3, cloud, png):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, height_ratios=[1.1, 1, 1.1])
    ax_acc, ax_zoom, ax_loss = axes
    for s in SEEDS:
        d = m3[s]
        ax_acc.plot(d["steps"], [100 * a for a in d["train_acc"]], color=COLORS[s], lw=1.4, ls="--")
        ax_acc.plot(d["steps"], [100 * a for a in d["test_acc"]], color=COLORS[s], lw=2)
        ax_zoom.plot(d["steps"], [100 * a for a in d["test_acc"]], color=COLORS[s], lw=2)
        c = cloud[s]
        ax_zoom.plot(c["steps"], [100 * a for a in c["test_acc"]], color=COLORS[s], lw=1, ls=":")
        ax_loss.plot(d["steps"], d["test_loss"], color=COLORS[s], lw=2)

    handles = [plt.Line2D([], [], color=COLORS[s], lw=2, label=f"seed {s}") for s in SEEDS]
    acc_styles = [plt.Line2D([], [], color="#52514e", lw=2, label="test (M3)"),
                  plt.Line2D([], [], color="#52514e", lw=1.4, ls="--", label="train (M3)")]
    ax_acc.axhline(95, color="#52514e", lw=0.8, ls=":")
    ax_acc.text(1.2, 96.5, "grokking threshold, test ≥ 95%", fontsize=8, color="#52514e", va="bottom")
    ax_acc.set_ylim(0, 108)
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title("less_data (15% of pairs, weight decay 1.0), 100,000 steps on the M3 CPU", fontsize=11)
    ax_acc.legend(handles=handles + acc_styles, fontsize=8, loc="center left", frameon=False, ncol=1)

    ax_zoom.axhline(100 * CHANCE, color="#52514e", lw=0.8, ls=":")
    ax_zoom.text(1.2, 100 * CHANCE + 0.08, "chance, 1/97", fontsize=8, color="#52514e", va="bottom")
    ax_zoom.set_ylim(0, 4)
    ax_zoom.set_ylabel("test accuracy (%), zoomed")
    ax_zoom.legend(handles=[plt.Line2D([], [], color="#52514e", lw=2, label="M3, 100k steps"),
                            plt.Line2D([], [], color="#52514e", lw=1, ls=":", label="cloud CPU, 60k steps (PR #2)")],
                   fontsize=8, loc="upper left", frameon=False)

    ax_loss.axhline(UNIFORM_LOSS, color="#52514e", lw=0.8, ls=":")
    ax_loss.text(1.2, UNIFORM_LOSS + 0.8, "ln(97) ≈ 4.57, uniform guess", fontsize=8, color="#52514e", va="bottom")
    ax_loss.set_ylim(bottom=0)
    ax_loss.set_ylabel("test loss (nats)")
    ax_loss.set_xlabel("training step (log scale)")
    ax_loss.set_xscale("log")
    ax_loss.set_xlim(1, STEPS * 1.15)
    for ax in axes:
        ax.grid(alpha=0.25, which="major")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def fmt_grok(r):
    return f"{r['grok']:,}" if r["grok"] is not None else f"did not grok within {STEPS:,} steps"


def report(sums, trends, checks, comps):
    print("| seed | memorisation step (train ≥ 99%) | max test acc | final test acc | final test loss | vs ln(97) ≈ 4.57 | grokking step (test ≥ 95%) |")
    print("|---|---|---|---|---|---|---|")
    for r in sums:
        print(f"| {r['seed']} | {r['memorised']:,} | {r['max_test']:.2%} (step {r['max_test_step']:,}) | {r['final_test']:.2%} | "
              f"{r['final_test_loss']:.2f} | {r['final_test_loss'] / UNIFORM_LOSS:.1f}× | {fmt_grok(r)} |")
    print("\nmax test acc from memorisation onward:", {r["seed"]: f"{r['max_test_after_mem']:.2%}" for r in sums})
    print("test acc range, steps >= 50,000:", {r["seed"]: f"{r['late_test_min']:.2%}–{r['late_test_max']:.2%}" for r in sums})
    print("\nruntime:", {r["seed"]: r["seconds"] for r in sums}, "recorded points:", {r["seed"]: r["n_recorded"] for r in sums})
    print("\ntest loss at (last recorded step at or before) each checkpoint:")
    print("| seed | " + " | ".join(f"{c:,}" for c in CHECKPOINTS) + " | peak (step) |")
    print("|---|" + "---|" * (len(CHECKPOINTS) + 1))
    for r, chk in zip(sums, checks):
        print(f"| {r['seed']} | " + " | ".join(f"{v:.2f}" for _, v in chk) + f" | {r['peak_test_loss']:.2f} ({r['peak_test_loss_step']:,}) |")
    print("recorded steps used:", [s for s, _ in checks[0]])
    print("\ntest-loss trend (OLS over window):")
    print("| seed | window | points | fitted change (nats) | % of window mean | trend |")
    print("|---|---|---|---|---|---|")
    for s, t in zip(SEEDS, trends):
        for (lo, hi), w in zip(WINDOWS, t):
            print(f"| {s} | {lo:,}–{hi:,} | {w['n']} | {w['change']:+.3f} | {w['pct']:+.2f}% | {w['label']} |")
    print("\nM3 vs cloud over the first 60k steps:")
    for c in comps:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}))
    print("\nmedian across seeds of final test loss:", round(statistics.median(r["final_test_loss"] for r in sums), 2))


def main():
    m3 = {s: load_m3(s) for s in SEEDS}
    cloud = load_cloud()
    notes = ROOT / "notes"
    write_csv(m3, cloud, notes / "less_data_long.csv")
    plot(m3, cloud, notes / "less_data_long.png")
    sums = [summary(s, m3[s]) for s in SEEDS]
    trends = [[window_trend(m3[s], lo, hi) for lo, hi in WINDOWS] for s in SEEDS]
    checks = [[at_or_before(m3[s], "test_loss", c) for c in CHECKPOINTS] for s in SEEDS]
    comps = [compare(s, m3[s], cloud[s]) for s in SEEDS]
    report(sums, trends, checks, comps)


if __name__ == "__main__":
    main()
