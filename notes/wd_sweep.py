"""Weight-decay sweep: collect grokking steps into notes/wd_sweep{.csv,_100k.csv,.png}.

    python notes/wd_sweep.py

Sweep runs are read from outputs/wd<W>__seed<S>/run.json (produced by
`python train/train.py --run baseline --seed S --set weight_decay=W --tag wd<W>__seed<S>`).
All of wd in {0.03, 0.1, 0.3, 0.5, 1.0} x seeds {0, 1, 2} must be present; a
missing run is an error. The wd = 0 rows are read from the committed
site/data/runs/no_decay.json (same M3, 4-dp rounded, not rerun). The committed
baseline.json is used only to check that the wd = 1.0 runs reproduce it.

Longer budget: wd in {0.03, 0.1, 0.3}, seed 0, 100,000 steps, read from
outputs/wd<W>__seed0__100k/run.json (produced by `python train/train.py --run baseline
--seed 0 --set weight_decay=W --set steps=100000 --tag wd<W>__seed0__100k`). A missing or
incomplete run is an error. Results go to notes/wd_sweep_100k.csv and onto the PNG; the
25k CSV is not affected. Both markdown tables are printed for the note.

Grokking step = the repo's definition: the first *recorded* step with
test_acc >= 0.95 (train/plot.py: milestones()["test_95"], first_step()).
"""

from __future__ import annotations

import bisect
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
from config import RUNS  # noqa: E402
from plot import first_step  # noqa: E402  (the repo's definition)

THRESH = 0.95
SWEEP_WD = [0.03, 0.1, 0.3, 0.5, 1.0]
SWEEP_SEEDS = [0, 1, 2]
NEAR_BUDGET_FRAC = 0.10  # a grok in the last 10% of the step budget is "near the budget"
LONG_WD = [0.03, 0.1, 0.3]
LONG_SEED = 0
LONG_STEPS = 100_000
CHECKPOINTS = [25_000, 50_000, 75_000]  # test acc at the last recorded step <= each of these


def row(source, wd, seed, steps, train_acc, test_acc, budget, seconds=None):
    g = first_step(steps, test_acc, THRESH)
    return {
        "source": source,
        "weight_decay": wd,
        "seed": seed,
        "grok_step": g if g is not None else "",
        "grokked": g is not None,
        "near_budget": g is not None and g >= budget * (1 - NEAR_BUDGET_FRAC),
        "step_budget": budget,
        "final_train_acc": round(train_acc[-1], 4),
        "final_test_acc": round(test_acc[-1], 4),
        "max_test_acc": round(max(test_acc), 4),
        "seconds": seconds if seconds is not None else "",
    }


def committed(name):
    """A committed site run as (config, per-seed curves sorted by seed)."""
    d = json.loads((ROOT / "site/data/runs" / f"{name}.json").read_text())
    c = d["config"]
    runs = [{"seed": c["seed"], **{k: d[k] for k in ("steps", "train_acc", "test_acc")}}] + d["seed_runs"]
    return c, sorted(runs, key=lambda r: r["seed"])


def sweep_run(wd, seed):
    """One sweep run's run.json, checked to be the baseline config apart from wd and seed."""
    f = ROOT / "outputs" / f"wd{wd}__seed{seed}" / "run.json"
    d = json.loads(f.read_text())  # a missing run is an error: the sweep must be complete
    expected = RUNS["baseline"].config.with_overrides(weight_decay=wd, seed=seed).to_dict()
    assert d["config"] == expected, f"{f}: config differs from baseline + wd/seed"
    assert d["runtime"]["device"] == "cpu", f
    return d


def reproduction():
    """Compare the rerun wd = 1.0 grokking steps with the committed baseline, per seed."""
    _, runs = committed("baseline")
    out = []
    for r in runs:
        d = sweep_run(1.0, r["seed"])
        new, old = first_step(d["steps"], d["test_acc"], THRESH), first_step(r["steps"], r["test_acc"], THRESH)
        same = d["steps"] == r["steps"] and [round(x, 4) for x in d["test_acc"]] == r["test_acc"]
        out.append(f"seed {r['seed']}: rerun {new} vs published {old} "
                   f"({'MATCH' if new == old else 'DIFFERENT'}; 4-dp test curve "
                   f"{'identical' if same else 'differs'})")
    return "\n".join(out)


def collect():
    rows = []
    c, runs = committed("no_decay")
    for r in runs:
        rows.append(row("existing:no_decay", c["weight_decay"], r["seed"], r["steps"], r["train_acc"],
                        r["test_acc"], c["steps"]))
    for wd in SWEEP_WD:
        for s in SWEEP_SEEDS:
            d = sweep_run(wd, s)
            c = d["config"]
            rows.append(row("sweep", wd, s, d["steps"], d["train_acc"], d["test_acc"], c["steps"],
                            d["runtime"]["seconds"]))
    return sorted(rows, key=lambda r: (r["weight_decay"], r["seed"]))


def long_run(wd):
    """One 100k run's run.json, checked to be baseline apart from wd and steps, and complete."""
    f = ROOT / "outputs" / f"wd{wd}__seed{LONG_SEED}__100k" / "run.json"
    d = json.loads(f.read_text())  # a missing run is an error
    expected = RUNS["baseline"].config.with_overrides(weight_decay=wd, seed=LONG_SEED, steps=LONG_STEPS).to_dict()
    assert d["config"] == expected, f"{f}: config differs from baseline + wd/steps"
    assert d["runtime"]["device"] == "cpu", f
    assert d["steps"][-1] == LONG_STEPS, f"{f}: stopped at step {d['steps'][-1]}"
    return d


def at_or_before(steps, values, step):
    """Value at the last recorded step <= step."""
    return values[bisect.bisect_right(steps, step) - 1]


def collect_long():
    rows = []
    for wd in LONG_WD:
        d = long_run(wd)
        r = row("long100k", wd, LONG_SEED, d["steps"], d["train_acc"], d["test_acc"], LONG_STEPS,
                d["runtime"]["seconds"])
        test = d["test_acc"]
        r["last_recorded_step"] = d["steps"][-1]
        r["max_test_acc_step"] = d["steps"][test.index(max(test))]
        after = [a for s, a in zip(d["steps"], test) if r["grokked"] and s >= r["grok_step"]]
        r["min_test_acc_after_grok"] = round(min(after), 4) if after else ""
        for c in CHECKPOINTS:
            r[f"step_recorded_at_{c // 1000}k"] = at_or_before(d["steps"], d["steps"], c)
            r[f"test_acc_at_{c // 1000}k"] = round(at_or_before(d["steps"], test, c), 4)
        # Same seed, config and code as the 25k run: where the two recording schedules
        # share a step, the curves should agree.
        short = sweep_run(wd, LONG_SEED)
        by_step = dict(zip(short["steps"], short["test_acc"]))
        common = [(by_step[s], a) for s, a in zip(d["steps"], test) if s in by_step]
        r["steps_shared_with_25k_run"] = len(common)
        r["last_shared_step"] = max(s for s in d["steps"] if s in by_step)
        r["max_abs_test_acc_diff_vs_25k_run"] = max(abs(x - y) for x, y in common)
        rows.append(r)
    return rows


def fmt_step(r):
    if not r["grokked"]:
        return f"did not grok within {r['step_budget']:,} steps"
    return f"{r['grok_step']:,}" + (" (near the budget)" if r["near_budget"] else "")


def markdown(rows):
    out = ["| weight decay | seed | source | grokking step (test acc ≥ 95%) | final train acc | final test acc |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        src = "this sweep" if r["source"] == "sweep" else f"committed `{r['source'].split(':')[1]}`"
        out.append(f"| {r['weight_decay']:g} | {r['seed']} | {src} | {fmt_step(r)} | "
                   f"{r['final_train_acc']:.2%} | {r['final_test_acc']:.2%} |")
    return "\n".join(out)


def markdown_long(rows):
    out = ["| weight decay | seed | grokking step (test acc ≥ 95%) | max test acc | at step | final train acc | final test acc |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['weight_decay']:g} | {r['seed']} | {fmt_step(r)} | {r['max_test_acc']:.2%} | "
                   f"{r['max_test_acc_step']:,} | {r['final_train_acc']:.2%} | {r['final_test_acc']:.2%} |")
    return "\n".join(out)


def plot(rows, long_rows, png):
    pos = [r for r in rows if r["weight_decay"] > 0]
    budget = max(r["step_budget"] for r in rows)
    # Two rows above the 100k line for "did not grok"; they are categories, not step values.
    row_100k, row_25k = LONG_STEPS * 1.8, LONG_STEPS * 3.4
    fig, ax = plt.subplots(figsize=(7.4, 5.8), layout="constrained")
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    def jx(r):  # small multiplicative x-jitter per seed so coincident points stay visible
        return r["weight_decay"] * (1 + 0.05 * (r["seed"] - 1))

    ax.axhspan(LONG_STEPS * 1.3, row_25k * 1.5, color="grey", alpha=0.08, lw=0, zorder=1)
    for seed in sorted({r["seed"] for r in pos}):
        g = [r for r in pos if r["seed"] == seed and r["grokked"]]
        n = [r for r in pos if r["seed"] == seed and not r["grokked"]]
        ax.scatter([jx(r) for r in g], [r["grok_step"] for r in g], color=colors[seed],
                   s=40, zorder=3, label=f"seed {seed}, 25k budget", alpha=0.85)
        ax.scatter([jx(r) for r in n], [row_25k] * len(n), color=colors[seed], marker="x",
                   s=60, zorder=3, linewidths=2)
    ax.scatter([], [], color="grey", marker="x", s=60, linewidths=2, label="no grok within 25k (colour = seed)")
    ax.scatter([jx(r) for r in long_rows], [r["grok_step"] if r["grokked"] else row_100k for r in long_rows],
               marker="D", s=90, facecolors="none", edgecolors=colors[LONG_SEED], linewidths=2, zorder=5,
               label=f"seed {LONG_SEED}, 100k budget")
    wds = sorted({r["weight_decay"] for r in pos})
    med_x, med_y = [], []
    for wd in wds:
        rs = [r for r in pos if r["weight_decay"] == wd]
        if all(r["grokked"] for r in rs):  # median only when every seed grokked (no extrapolation)
            med_x.append(wd)
            med_y.append(statistics.median(r["grok_step"] for r in rs))
    ax.plot(med_x, med_y, color="black", marker="_", markersize=22, mew=2.5, lw=1, zorder=4,
            label="median (all 3 seeds grokked, 25k)")
    for b in (budget, LONG_STEPS):
        ax.axhline(b, color="grey", lw=0.8, ls="--")
        ax.text(0.01, b * 1.04, f"{b:,}-step budget", transform=ax.get_yaxis_transform(), fontsize=8,
                color="dimgrey", va="bottom")
    ax.axhspan(budget * (1 - NEAR_BUDGET_FRAC), budget, color="grey", alpha=0.15, lw=0, zorder=1,
               label="last 10% of 25k budget")
    nd = [r for r in rows if r["weight_decay"] == 0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3_000, row_25k * 1.5)
    ax.set_xlim(wds[0] * 0.75, wds[-1] * 1.3)
    ax.set_xticks(wds)
    ax.set_xticklabels([f"{w:g}" for w in wds])
    ax.minorticks_off()
    ticks = [5_000, 10_000, 25_000, 50_000, LONG_STEPS]
    ax.set_yticks(ticks + [row_100k, row_25k])
    ax.set_yticklabels([f"{t:,}" for t in ticks] + ["no grok\nby 100,000", "no grok\nby 25,000"], fontsize=8)
    ax.set_xlabel("weight decay (log scale; seeds slightly offset). All runs: Apple M3 CPU")
    ax.set_ylabel("grokking step (first recorded test acc ≥ 95%)")
    ax.set_title("(a + b) mod 97, 30% train data: grokking step vs weight decay\n"
                 f"wd = 0 (committed no_decay, not on this log axis): {sum(not r['grokked'] for r in nd)}/{len(nd)} "
                 f"seeds did not grok within {budget:,} steps", fontsize=10)
    ax.grid(alpha=0.3)
    fig.legend(fontsize=8, loc="outside lower center", ncols=3)
    fig.savefig(png, dpi=130)


def main():
    rows = collect()
    notes = ROOT / "notes"
    with open(notes / "wd_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    long_rows = collect_long()
    with open(notes / "wd_sweep_100k.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    plot(rows, long_rows, notes / "wd_sweep.png")
    print(markdown(rows))
    print("\nLonger budget (100,000 steps):\n" + markdown_long(long_rows))
    print("\nwd = 1.0 reproduction check:\n" + reproduction())


if __name__ == "__main__":
    main()
