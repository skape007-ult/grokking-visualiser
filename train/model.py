"""A tiny transformer following Nanda et al. 2023.

No LayerNorm, learned positional embeddings, untied embedding / unembedding,
ReLU MLP. Initialisation mirrors Nanda's reference code (Gaussian scaled by
1/sqrt(fan_in), zero biases).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RunConfig


def _param(*shape, scale):
    return nn.Parameter(torch.randn(*shape) * scale)


class Block(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        d, h, m = cfg.d_model, cfg.n_heads, cfg.d_mlp
        self.n_heads, self.d_head = h, d // h
        self.W_Q = _param(d, d, scale=1 / math.sqrt(d))
        self.W_K = _param(d, d, scale=1 / math.sqrt(d))
        self.W_V = _param(d, d, scale=1 / math.sqrt(d))
        self.W_O = _param(d, d, scale=1 / math.sqrt(d))
        self.W_in = _param(d, m, scale=1 / math.sqrt(d))
        self.b_in = nn.Parameter(torch.zeros(m))
        self.W_out = _param(m, d, scale=1 / math.sqrt(m))
        self.b_out = nn.Parameter(torch.zeros(d))
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool)), persistent=False
        )

    def forward(self, x, last_only: bool = False):
        # last_only: compute only the final position's output. Exact (not an
        # approximation) for the last block, since only '=' feeds the logits,
        # and ~4x cheaper.
        B, T, D = x.shape
        Tq = 1 if last_only else T
        split = lambda t, n: t.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        q = split(x[:, T - Tq:] @ self.W_Q, Tq)
        k, v = split(x @ self.W_K, T), split(x @ self.W_V, T)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.mask[T - Tq:T, :T], float("-inf")).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, Tq, D)
        x = x[:, T - Tq:] + z @ self.W_O
        x = x + F.relu(x @ self.W_in + self.b_in) @ self.W_out + self.b_out
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        vocab = cfg.p + 2
        self.W_E = _param(vocab, cfg.d_model, scale=1 / math.sqrt(cfg.d_model))
        self.W_pos = _param(cfg.n_ctx, cfg.d_model, scale=1 / math.sqrt(cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        # Unembedding maps to the p possible answers only.
        self.W_U = _param(cfg.d_model, cfg.p, scale=1 / math.sqrt(vocab))

    def forward(self, tokens):
        x = self.W_E[tokens] + self.W_pos[: tokens.shape[1]]
        for i, block in enumerate(self.blocks):
            x = block(x, last_only=i == len(self.blocks) - 1)
        return x[:, -1] @ self.W_U  # logits from the final '=' position

    def number_embeddings(self, p: int) -> torch.Tensor:
        """Embedding rows for the p number tokens (p x d_model)."""
        return self.W_E[:p].detach()
