"""Write site/data/manifest.json and site/data/runs/<id>.json from outputs/.

    python train/export.py

This is the only code that writes site/data/. Never hand-edit those files:
fix this generator (or the run) and re-export.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from config import SWEEP

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SITE_DATA = ROOT / "site" / "data"
DECIMALS = 4


def r(x):
    """Round floats (recursively) to DECIMALS places; reject NaN/Inf."""
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("non-finite value in export")
        return round(x, DECIMALS)
    if isinstance(x, list):
        return [r(v) for v in x]
    if isinstance(x, dict):
        return {k: r(v) for k, v in x.items()}
    return x


def export():
    runs_dir = SITE_DATA / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True)

    manifest = {"p": None, "operation": None, "runs": []}
    for spec in SWEEP:
        cfg = spec.config
        run = json.loads((OUTPUTS / cfg.id / "run.json").read_text())
        proj = json.loads((OUTPUTS / cfg.id / "projected.json").read_text())
        if run["config"] != cfg.to_dict():
            raise SystemExit(f"{cfg.id}: outputs/ were produced by a different config; re-train")

        data = {
            "config": run["config"],
            "steps": run["steps"],
            "train_acc": r(run["train_acc"]),
            "test_acc": r(run["test_acc"]),
            "train_loss": r(run["train_loss"]),
            "test_loss": r(run["test_loss"]),
            "embedding_frames": [
                {"step": s, "coords": r(c)}
                for s, c in zip(proj["embedding_steps"], proj["coords"])
            ],
        }
        rel = f"runs/{cfg.id}.json"
        (SITE_DATA / rel).write_text(json.dumps(data, separators=(",", ":")))

        manifest["p"], manifest["operation"] = cfg.p, cfg.operation
        manifest["runs"].append(
            {
                "id": cfg.id,
                "label": spec.label,
                "description": spec.description,
                "weight_decay": cfg.weight_decay,
                "train_frac": cfg.train_frac,
                "file": rel,
            }
        )
        print(f"wrote site/data/{rel} ({(SITE_DATA / rel).stat().st_size / 1024:.0f} KB)")

    (SITE_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote site/data/manifest.json")


if __name__ == "__main__":
    export()
