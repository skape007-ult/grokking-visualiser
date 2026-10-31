"""Check site/data/ against the Data Contract. Exits non-zero on any failure.

    python train/validate.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SITE_DATA = Path(__file__).resolve().parent.parent / "site" / "data"
MAX_BYTES = 200 * 1024
METRICS = ("train_acc", "test_acc", "train_loss", "test_loss")
EXPECTED_ORDER = ["baseline", "no_decay", "more_data", "less_data"]

errors: list[str] = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
    return cond


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def max_decimals_ok(x):
    return round(x, 4) == x


def validate_run(entry, p):
    rid = entry["id"]
    path = SITE_DATA / entry["file"]
    if not check(path.is_file(), f"{rid}: missing {entry['file']}"):
        return
    check(path.stat().st_size <= MAX_BYTES, f"{rid}: {path.stat().st_size} bytes > {MAX_BYTES}")
    d = json.loads(path.read_text())

    cfg = d.get("config", {})
    for k in ("p", "weight_decay", "train_frac", "seed", "steps", "lr", "d_model"):
        check(k in cfg, f"{rid}: config missing {k!r}")
    check(cfg.get("id") == rid, f"{rid}: config.id is {cfg.get('id')!r}")
    check(cfg.get("p") == p, f"{rid}: config.p != manifest p")
    check(cfg.get("weight_decay") == entry["weight_decay"], f"{rid}: weight_decay mismatch with manifest")
    check(cfg.get("train_frac") == entry["train_frac"], f"{rid}: train_frac mismatch with manifest")

    steps = d["steps"]
    check(all(isinstance(s, int) for s in steps), f"{rid}: non-integer step")
    check(steps == sorted(set(steps)), f"{rid}: steps not strictly increasing")
    check(steps[0] == 1 and steps[-1] == cfg.get("steps"), f"{rid}: steps must span 1..final step")
    for m in METRICS:
        arr = d[m]
        check(len(arr) == len(steps), f"{rid}: len({m})={len(arr)} != len(steps)={len(steps)}")
        check(all(is_num(v) for v in arr), f"{rid}: {m} has NaN/Inf/non-number")
        check(all(max_decimals_ok(v) for v in arr), f"{rid}: {m} not rounded to 4 dp")
        if m.endswith("acc"):
            check(all(0 <= v <= 1 for v in arr), f"{rid}: {m} outside [0, 1]")
        else:
            check(all(v >= 0 for v in arr), f"{rid}: {m} negative")

    frames = d["embedding_frames"]
    fsteps = [f["step"] for f in frames]
    step_set = set(steps)
    check(len(frames) >= 2, f"{rid}: fewer than 2 embedding frames")
    check(fsteps == sorted(set(fsteps)), f"{rid}: frame steps not strictly ascending")
    check(all(s in step_set for s in fsteps), f"{rid}: frame step not in steps")
    check(fsteps[0] == steps[0] and fsteps[-1] == steps[-1], f"{rid}: frames must include first and final step")
    # Guards against snapshots aliasing the live weights (all frames equal).
    check(frames[0]["coords"] != frames[-1]["coords"], f"{rid}: first and final frames are identical")
    for f in frames:
        c = f["coords"]
        check(len(c) == p, f"{rid}: frame {f['step']} has {len(c)} points, expected {p}")
        check(
            all(len(xy) == 2 and all(is_num(v) and max_decimals_ok(v) for v in xy) for xy in c),
            f"{rid}: frame {f['step']} has bad coords",
        )


def main():
    manifest = json.loads((SITE_DATA / "manifest.json").read_text())
    p = manifest.get("p")
    check(isinstance(p, int) and p > 1, "manifest: bad p")
    check(isinstance(manifest.get("operation"), str), "manifest: missing operation")
    ids = [e["id"] for e in manifest["runs"]]
    check(ids == EXPECTED_ORDER, f"manifest: run order {ids} != {EXPECTED_ORDER}")
    for e in manifest["runs"]:
        for k in ("id", "label", "description", "weight_decay", "train_frac", "file"):
            check(k in e, f"manifest entry {e.get('id')}: missing {k!r}")
        check(e["file"] == f"runs/{e['id']}.json", f"{e['id']}: file path should be runs/{e['id']}.json")
        validate_run(e, p)

    # No orphan run files that the manifest doesn't reference.
    on_disk = {f"runs/{f.name}" for f in (SITE_DATA / "runs").glob("*.json")}
    check(on_disk == {e["file"] for e in manifest["runs"]}, f"orphan/missing run files: {on_disk ^ {e['file'] for e in manifest['runs']}}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: manifest + {len(ids)} runs valid ({', '.join(ids)})")


if __name__ == "__main__":
    main()
