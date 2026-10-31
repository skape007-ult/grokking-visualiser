"""Train one run: record metrics at log-spaced steps and snapshot embeddings.

Usage (from the repo root):
    python train/train.py --run baseline
    python train/train.py --run baseline --set train_frac=0.25 --tag attempt2

Writes outputs/<tag or id>/run.json (config + metrics) and embeddings.npy
(raw number-token embedding snapshots, n_snapshots x p x d_model).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import RUNS, RunConfig, parse_override
from data import make_dataset
from model import Transformer

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The embedding lookup's backward pass (index_put with accumulate) is
    # otherwise nondeterministic on multithreaded CPU (~1e-8 differences that
    # compound over thousands of steps). This makes runs bit-for-bit repeatable.
    torch.use_deterministic_algorithms(True)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_steps(final: int, n: int) -> list[int]:
    """n log-spaced, deduplicated integer steps from 1 to final (inclusive).

    Rounding to integers collapses many early points (1, 1, 1, 2, ...), so the
    raw count is increased until n distinct steps survive deduplication.
    """
    n = min(n, final)
    k = n
    while True:
        steps = set(np.round(np.logspace(0, np.log10(final), k)).astype(int).tolist()) | {1, final}
        if len(steps) >= n:
            return sorted(steps)
        k += 1


def snapshot_steps(record: list[int], n: int) -> list[int]:
    """n log-spaced steps chosen from `record` (includes first and last)."""
    rec = np.array(record)
    n = min(n, len(record))
    k = n
    while True:
        targets = np.logspace(0, np.log10(record[-1]), k)
        picked = {int(rec[np.argmin(np.abs(np.log(rec) - np.log(t)))]) for t in targets}
        picked |= {record[0], record[-1]}
        if len(picked) >= n:
            return sorted(picked)
        k += 1


def loss_fn(logits, y, dtype):
    return F.cross_entropy(logits.to(dtype), y)


@torch.no_grad()
def evaluate(model, x, y, dtype):
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, y, dtype).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return loss, acc


def train(cfg: RunConfig, out_name: str | None = None) -> Path:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    loss_dtype = getattr(torch, cfg.loss_dtype)
    if device.type == "mps" and loss_dtype == torch.float64:
        loss_dtype = torch.float32  # MPS has no float64; recorded below.

    train_x, train_y, test_x, test_y = (t.to(device) for t in make_dataset(cfg))
    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / cfg.warmup_steps, 1.0) if cfg.warmup_steps else 1.0
    )

    record = log_steps(cfg.steps, cfg.n_record)
    snaps = set(snapshot_steps(record, cfg.n_snapshots))
    record_set = set(record)
    metrics = {k: [] for k in ("steps", "train_loss", "test_loss", "train_acc", "test_acc")}
    emb_steps, emb = [], []

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        loss = loss_fn(model(train_x), train_y, loss_dtype)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step in record_set:
            tr_l, tr_a = evaluate(model, train_x, train_y, loss_dtype)
            te_l, te_a = evaluate(model, test_x, test_y, loss_dtype)
            for k, v in zip(metrics, (step, tr_l, te_l, tr_a, te_a)):
                metrics[k].append(v)
            if step in snaps:
                emb_steps.append(step)
                emb.append(model.number_embeddings(cfg.p).float().cpu().numpy())
            if step in snaps or step == cfg.steps:
                print(
                    f"[{cfg.id}] step {step:>6}  train {tr_a:6.1%} ({tr_l:.3g})  "
                    f"test {te_a:6.1%} ({te_l:.3g})  {time.time() - t0:5.0f}s",
                    flush=True,
                )

    out = OUTPUTS / (out_name or cfg.id)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", np.stack(emb))
    (out / "run.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "runtime": {
                    "device": str(device),
                    "loss_dtype_used": str(loss_dtype).removeprefix("torch."),
                    "torch": torch.__version__,
                    "seconds": round(time.time() - t0, 1),
                },
                **metrics,
                "embedding_steps": emb_steps,
            },
            indent=1,
        )
    )
    print(f"[{cfg.id}] wrote {out} in {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline", choices=sorted(RUNS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field (for exploratory attempts)")
    ap.add_argument("--tag", help="output directory name (defaults to the run id)")
    args = ap.parse_args()
    cfg = RUNS[args.run].config
    for a in args.set:
        cfg = parse_override(cfg, a)
    train(cfg, args.tag)


if __name__ == "__main__":
    main()
