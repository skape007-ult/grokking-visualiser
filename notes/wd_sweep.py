"""Weight-decay sweep: collect grokking steps into notes/wd_sweep.{csv,png}.

    python notes/wd_sweep.py

Sweep runs are read from outputs/wd<W>__seed<S>/run.json (produced by
`python train/train.py --run baseline --seed S --set weight_decay=W --tag wd<W>__seed<S>`).
Context rows (wd = 1.0 = `baseline`, wd = 0 = `no_decay`) are read from the
committed site/data/runs/*.json (existing results, 4-dp rounded, not rerun).

Grokking step = the repo's definition: the first *recorded* step with
test_acc >= 0.95 (train/plot.py: milestones()["test_95"], first_step()).
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot import first_step  # noqa: E402  (the repo's definition)

THRESH = 0.95
SWEEP_WD = [0.03, 0.1, 0.3, 0.5]
SWEEP_SEEDS = [0, 1, 2]


def row(source, wd, seed, steps, train_acc, test_acc, budget, seconds=None):
    g = first_step(steps, test_acc, THRESH)
    return {
        "source": source,
        "weight_decay": wd,
        "seed": seed,
        "grok_step": g if g is not None else "",
        "grokked": g is not None,
        "step_budget": budget,
        "final_train_acc": round(train_acc[-1], 4),
        "final_test_acc": round(test_acc[-1], 4),
        "max_test_acc": round(max(test_acc), 4),
        "seconds": seconds if seconds is not None else "",
    }


def collect():
    rows = []
    for name, src in (("no_decay", "existing:no_decay"), ("baseline", "existing:baseline")):
        d = json.loads((ROOT / "site/data/runs" / f"{name}.json").read_text())
        c = d["config"]
        runs = [{"seed": c["seed"], **{k: d[k] for k in ("steps", "train_acc", "test_acc")}}] + d["seed_runs"]
        for r in sorted(runs, key=lambda r: r["seed"]):
            rows.append(row(src, c["weight_decay"], r["seed"], r["steps"], r["train_acc"], r["test_acc"], c["steps"]))
    for wd in SWEEP_WD:
        for s in SWEEP_SEEDS:
            f = ROOT / "outputs" / f"wd{wd}__seed{s}" / "run.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            c = d["config"]
            assert c["weight_decay"] == wd and c["seed"] == s, f
            rows.append(row("sweep", wd, s, d["steps"], d["train_acc"], d["test_acc"], c["steps"],
                            d["runtime"]["seconds"]))
    return sorted(rows, key=lambda r: (r["weight_decay"], r["seed"]))


def fmt_step(r):
    return f"{r['grok_step']:,}" if r["grokked"] else f"did not grok within {r['step_budget']:,} steps"


def markdown(rows):
    out = ["| weight decay | seed | source | grokking step (test acc ≥ 95%) | final train acc | final test acc |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        src = "sweep (this PR)" if r["source"] == "sweep" else f"existing `{r['source'].split(':')[1]}`"
        out.append(f"| {r['weight_decay']:g} | {r['seed']} | {src} | {fmt_step(r)} | "
                   f"{r['final_train_acc']:.2%} | {r['final_test_acc']:.2%} |")
    return "\n".join(out)


def plot(rows, png):
    pos = [r for r in rows if r["weight_decay"] > 0]
    budget = max(r["step_budget"] for r in rows)
    top = budget * 1.6  # band above the budget for "did not grok"
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    for seed in sorted({r["seed"] for r in pos}):
        g = [r for r in pos if r["seed"] == seed and r["grokked"]]
        n = [r for r in pos if r["seed"] == seed and not r["grokked"]]
        ax.scatter([r["weight_decay"] for r in g], [r["grok_step"] for r in g], color=colors[seed],
                   s=40, zorder=3, label=f"seed {seed}", alpha=0.85)
        ax.scatter([r["weight_decay"] for r in n], [top] * len(n), color=colors[seed], marker="x",
                   s=60, zorder=3, linewidths=2)
    wds = sorted({r["weight_decay"] for r in pos})
    med_x, med_y = [], []
    for wd in wds:
        rs = [r for r in pos if r["weight_decay"] == wd]
        if all(r["grokked"] for r in rs):  # median only when every seed grokked (no extrapolation)
            med_x.append(wd)
            med_y.append(statistics.median(r["grok_step"] for r in rs))
    ax.plot(med_x, med_y, color="black", marker="_", markersize=22, mew=2.5, lw=1, zorder=4,
            label="median (all seeds grokked)")
    ax.axhline(budget, color="grey", lw=0.8, ls="--")
    ax.text(wds[0], budget * 1.05, f"step budget ({budget:,})", fontsize=8, color="grey", va="bottom")
    ax.text(wds[0], top * 1.12, "× = did not grok within budget", fontsize=8, va="bottom")
    nd = [r for r in rows if r["weight_decay"] == 0]
    ax.text(0.99, 0.02, f"wd = 0 (existing no_decay): {sum(not r['grokked'] for r in nd)}/{len(nd)} "
            f"seeds did not grok within {budget:,} steps", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=top * 1.6)
    ax.set_xticks(wds)
    ax.set_xticklabels([f"{w:g}" for w in wds])
    ax.set_xlabel("weight decay (log scale); wd = 1.0 points are the existing baseline")
    ax.set_ylabel("grokking step (first recorded test acc ≥ 95%)")
    ax.set_title("(a + b) mod 97, 30% train data: grokking step vs weight decay", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def main():
    rows = collect()
    notes = ROOT / "notes"
    with open(notes / "wd_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, notes / "wd_sweep.png")
    print(markdown(rows))


if __name__ == "__main__":
    main()
