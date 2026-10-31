"""Modular arithmetic dataset with a seeded train/test split."""

from __future__ import annotations

import numpy as np
import torch

from config import RunConfig


def make_dataset(cfg: RunConfig):
    """Return (train_x, train_y, test_x, test_y) as CPU tensors.

    Vocabulary has p + 2 tokens: numbers 0..p-1, then op (= p) and '=' (= p+1).
    Each input is [a, op, b, =]; the label is (a + b) mod p, an index into the
    p possible answers (the model's output logits range over answers only).
    """
    p = cfg.p
    op_tok, eq_tok = p, p + 1
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    x = np.stack([a, np.full_like(a, op_tok), b, np.full_like(a, eq_tok)], axis=1)
    y = (a + b) % p

    # The split derives only from the seed, so it is regenerable from the config.
    perm = np.random.default_rng(cfg.seed).permutation(p * p)
    n_train = int(round(cfg.train_frac * p * p))
    tr, te = perm[:n_train], perm[n_train:]

    def t(arr):
        return torch.as_tensor(arr, dtype=torch.long)

    return t(x[tr]), t(y[tr]), t(x[te]), t(y[te])
