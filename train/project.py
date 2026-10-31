"""Project each run's embedding snapshots into 2D using a fixed PCA basis.

    python train/project.py baseline no_decay more_data less_data

Reads outputs/<id>/embeddings.npy, writes outputs/<id>/projected.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def fit_pca(X: np.ndarray, k: int = 2):
    """Return (mean, components[k, d]) of X via SVD, with deterministic signs."""
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = Vt[:k]
    # SVD signs are arbitrary; flip each component so its largest-magnitude
    # loading is positive, so re-runs never mirror the plot.
    signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
    return mean, comps * signs[:, None]


def project_run(run_id: str) -> Path:
    out = OUTPUTS / run_id
    snaps = np.load(out / "embeddings.npy")  # (n_snapshots, p, d_model)
    run = json.loads((out / "run.json").read_text())

    # The basis is fitted ONCE, on the final snapshot only, and every snapshot
    # is projected with that same mean and components. Fitting a PCA per frame
    # would let the axes rotate/flip from frame to frame, and the animation
    # would show the axes moving rather than the embeddings.
    mean, comps = fit_pca(snaps[-1])
    coords = (snaps - mean) @ comps.T  # (n_snapshots, p, 2)

    final_centered = snaps[-1] - mean
    explained = float((final_centered @ comps.T).var(0).sum() / final_centered.var(0).sum())
    result = {
        "embedding_steps": run["embedding_steps"],
        "coords": coords.tolist(),
        "pca_explained_variance_final": explained,
    }
    path = out / "projected.json"
    path.write_text(json.dumps(result))
    print(f"[{run_id}] projected {len(coords)} frames; 2 PCs explain {explained:.1%} of final variance")
    return path


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        project_run(rid)
