"""Diagnostic plots and milestone summaries from raw outputs/<name>/run.json.

    python train/plot.py summary outputs/baseline [outputs/...]
    python train/plot.py curve   outputs/baseline notes/plots/phase0_baseline.png
    python train/plot.py sweep   notes/plots/phase1_sweep.png      (every run x seed in config.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def first_step(steps, values, threshold):
    """First recorded step at which value >= threshold (None if never)."""
    return next((s for s, v in zip(steps, values) if v >= threshold), None)


def milestones(run: dict) -> dict:
    s = run["steps"]
    return {
        "train_99": first_step(s, run["train_acc"], 0.99),
        "test_50": first_step(s, run["test_acc"], 0.50),
        "test_95": first_step(s, run["test_acc"], 0.95),
        "final_train_acc": run["train_acc"][-1],
        "final_test_acc": run["test_acc"][-1],
        "final_train_loss": run["train_loss"][-1],
        "final_test_loss": run["test_loss"][-1],
        "seconds": run.get("runtime", {}).get("seconds"),
    }


def _axes_curve(ax, run, title):
    ax.plot(run["steps"], run["train_acc"], label="train", color="#1f77b4")
    ax.plot(run["steps"], run["test_acc"], label="test", color="#d62728", linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left")


def curve(run_dir, out_png):
    run = load(run_dir)
    c = run["config"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    _axes_curve(a1, run, f"{c['id']}: (a+b) mod {c['p']}, wd={c['weight_decay']}, frac={c['train_frac']}")
    a2.plot(run["steps"], run["train_loss"], label="train", color="#1f77b4")
    a2.plot(run["steps"], run["test_loss"], label="test", color="#d62728", linestyle="--")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("step (log scale)")
    a2.set_ylabel("cross-entropy loss")
    a2.set_title("loss", fontsize=10)
    a2.grid(alpha=0.3)
    a2.legend()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


def sweep(out_png):
    """One panel per condition: base seed bold, other seeds as faint lines."""
    from config import SWEEP, seed_configs

    outputs = Path(__file__).resolve().parent.parent / "outputs"
    fig, axes = plt.subplots(1, len(SWEEP), figsize=(4.2 * len(SWEEP), 3.8), sharey=True)
    for ax, spec in zip(axes, SWEEP):
        base, *others = [load(outputs / c.output_name) for c in seed_configs(spec)]
        for o in others:
            ax.plot(o["steps"], o["train_acc"], color="#1f77b4", alpha=0.3, lw=1)
            ax.plot(o["steps"], o["test_acc"], color="#d62728", alpha=0.3, lw=1, linestyle="--")
        c = base["config"]
        _axes_curve(ax, base, f"{c['id']}  (wd={c['weight_decay']}, frac={c['train_frac']})")
    fig.suptitle("bold: seed 0; faint: other seeds", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


def main():
    cmd, *args = sys.argv[1:]
    if cmd == "summary":
        for d in args:
            print(d, json.dumps(milestones(load(d))))
    elif cmd == "curve":
        curve(*args)
    elif cmd == "sweep":
        sweep(args[0])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
