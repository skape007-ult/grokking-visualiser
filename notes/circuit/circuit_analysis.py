"""Restricted / excluded logits over training, as pre-registered (v2) in
notes/prereg_circuit.md. Section numbers (§) refer to that file.

Reads outputs/<tag>/run.json and checkpoints.pt for each seed (tag defaults to
circuit__seed<S>) and writes to notes/circuit/ (or --out):

    circuit_repro.csv           §2a grokking step per seed vs README (M3)
    circuit_keyfreqs.csv        key sets (W_E rule), logit top-5, overlaps
    circuit_final.csv           step 25,000: every condition (key, 100 random, harder sets)
    circuit_over_training.csv   every checkpoint: key + 5 tracked random sets
    circuit_events.csv          milestones, P3/P4 steps, P5 events, effect sizes
    circuit_predictions.csv     every pre-registered check per seed (pass/fail + numbers)
    circuit_over_training.png

    .venv/bin/python notes/circuit/circuit_analysis.py --seeds 0 1 2 3 4 --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
from config import RunConfig  # noqa: E402
from data import make_dataset  # noqa: E402
from model import Transformer  # noqa: E402

P = 97
LN_P = math.log(P)
NF = (P - 1) // 2                        # 48 frequencies
README_GROK = {0: 9868, 1: 5785, 2: 4216}  # M3, README.md
M3_KEYS = {0: [3, 10, 12, 20, 32], 1: [14, 27, 29, 33, 34], 2: [1, 9, 18, 34, 41]}  # notes/fourier.md
TOP = 5
N_RANDOM = 100
N_TRACK = 5
GRID_DLN = 0.05                          # §3 steepest: resampling spacing in ln t
FLOOR = {"ce": 0.1, "energy": 0.05}      # §3 undefined-event floors


# ============================================================ basics

def first_step(steps, values, ok, start=None):
    for s, v in zip(steps, values):
        if (start is None or s >= start) and ok(v):
            return int(s)
    return None


def grid_inputs(p=P):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    x = np.stack([a, np.full_like(a, p), b, np.full_like(a, p + 1)], 1)
    return a, b, torch.as_tensor(x, dtype=torch.long), (a + b) % p


def split_idx(cfg: RunConfig):
    """Grid row indices of the train / test sets, in make_dataset's order."""
    perm = np.random.default_rng(cfg.seed).permutation(cfg.p * cfg.p)
    n_train = int(round(cfg.train_frac * cfg.p * cfg.p))
    return perm[:n_train], perm[n_train:]


def input_basis(p=P):
    """Columns: constant, then cos_k, sin_k (k = 1..48) at columns 2k-1, 2k; unit norm over the grid."""
    _, _, _, s = grid_inputs(p)
    cols = [np.ones(p * p)]
    for k in range(1, NF + 1):
        cols += [np.cos(2 * np.pi * k * s / p), np.sin(2 * np.pi * k * s / p)]
    F = np.stack(cols, 1)
    return F / np.linalg.norm(F, axis=0, keepdims=True)


def fcols(freqs):
    return [c for k in freqs for c in (2 * k - 1, 2 * k)]


def ce(L, y):
    z = L - L.max(1, keepdims=True)
    return float((np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]).mean())


def acc(L, y):
    return float((L.argmax(1) == y).mean())


def ce_cal(L, y, iters=48):
    """min over tau of CE(L / tau): golden-section search on log(1/tau) in [-12, 12].

    CE(beta L) is convex in beta, hence unimodal in log beta; beta -> 0 gives ln p.
    """
    f = lambda lb: ce(L * math.exp(lb), y)
    lo, hi = -12.0, 12.0
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
    return min(f1, f2, f(lo), f(hi), LN_P)


def we_spectrum(W):
    """Per-frequency energy of mean-centred W (p x d), as notes/fourier_analysis.py."""
    p = W.shape[0]
    x = np.arange(p)
    ks = np.arange(1, NF + 1)
    C = np.cos(2 * np.pi * np.outer(ks, x) / p)
    S = np.sin(2 * np.pi * np.outer(ks, x) / p)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    Wc = W - W.mean(0)
    e = ((C @ Wc) ** 2).sum(1) + ((S @ Wc) ** 2).sum(1)
    assert np.isclose(e.sum(), (Wc ** 2).sum(), rtol=1e-4)
    return e


def logit_spectrum(L, F):
    coef = F.T @ L
    return np.array([(coef[2 * k - 1] ** 2 + coef[2 * k] ** 2).sum() for k in range(1, NF + 1)])


# ============================================================ restricted / excluded

class Ctx:
    """Per-seed fixed objects: basis, rows, QR of the test-row design per frequency set."""

    def __init__(self, F, tr, te, y):
        self.F, self.tr, self.te, self.y = F, tr, te, y
        self.ytr, self.yte = y[tr], y[te]
        self._qr = {}

    def q_test(self, freqs):
        key = tuple(freqs)
        if key not in self._qr:
            A = self.F[self.te][:, [0] + fcols(freqs)]
            self._qr[key] = np.linalg.qr(A)[0]
        return self._qr[key]


def m_restricted_test(L, ctx, freqs, cal=True):
    """Primary restricted logits: least-squares fit on test rows only (§3)."""
    Q = ctx.q_test(freqs)
    Lt = L[ctx.te]
    R = Q @ (Q.T @ Lt)
    out = {"restricted_test_acc": acc(R, ctx.yte), "restricted_test_ce": ce(R, ctx.yte)}
    if cal:
        out["restricted_test_cecal"] = ce_cal(R, ctx.yte)
    return out


def m_restricted_grid(L, ctx, freqs):
    """Secondary, LEAKING: full-grid projection onto const + S (§3)."""
    cols = [0] + fcols(freqs)
    R = ctx.F[:, cols] @ (ctx.F[:, cols].T @ L)
    return {"restrictedgrid_test_acc": acc(R[ctx.te], ctx.yte), "restrictedgrid_test_ce": ce(R[ctx.te], ctx.yte),
            "restrictedgrid_all_acc": acc(R, ctx.y)}


def proj_S(L, ctx, freqs):
    cols = fcols(freqs)
    if not cols:
        return np.zeros_like(L)
    return ctx.F[:, cols] @ (ctx.F[:, cols].T @ L)


def m_excluded(L, ctx, freqs, cal=True, PS=None):
    PS = proj_S(L, ctx, freqs) if PS is None else PS
    X = (L - PS)[ctx.tr]
    out = {"excluded_train_acc": acc(X, ctx.ytr), "excluded_train_ce": ce(X, ctx.ytr)}
    if cal:
        out["excluded_train_cecal"] = ce_cal(X, ctx.ytr)
    return out


def margin_share(L, ctx, freqs, PS=None):
    """phi_S on train rows (§3)."""
    PS = proj_S(L, ctx, freqs) if PS is None else PS
    r = np.arange(len(ctx.tr))
    Lt, Pt = L[ctx.tr], PS[ctx.tr]
    num = (Pt[r, ctx.ytr] - Pt.mean(1)).mean()
    den = (Lt[r, ctx.ytr] - Lt.mean(1)).mean()
    return float(num / den)


def all_measures(L, ctx, freqs, cal=True):
    PS = proj_S(L, ctx, freqs)
    return {**m_restricted_test(L, ctx, freqs, cal), **m_restricted_grid(L, ctx, freqs),
            **m_excluded(L, ctx, freqs, cal, PS), "margin_share": margin_share(L, ctx, freqs, PS)}


# ============================================================ events (§3)

def events(steps, m, t_mem, end_step, direction, kind):
    s = np.asarray(steps, float)
    v = np.asarray(m, float)
    sel = (s >= t_mem) & (s <= end_step)
    s, v = s[sel], v[sel]
    delta = v[-1] - v[0]
    if abs(delta) < FLOOR[kind] or np.sign(delta) != direction:
        return {"defined": False, "onset": None, "completion": None, "steepest": None, "delta": delta}
    prog = (v - v[0]) / delta
    onset = next((int(t) for t, q in zip(s, prog) if q >= 0.10), None)
    completion = None
    for i in range(len(s)):
        if (prog[i:] >= 0.90).all():
            completion = int(s[i])
            break
    lg = np.arange(math.log(s[0]), math.log(s[-1]) + 1e-12, GRID_DLN)
    vg = np.interp(lg, np.log(s), v)
    d = direction * np.diff(vg)
    j = int(np.argmax(d))
    steepest = math.exp((lg[j] + lg[j + 1]) / 2)
    return {"defined": True, "onset": onset, "completion": completion, "steepest": steepest, "delta": delta}


def phase(t, t10, t99):
    if t is None:
        return "undefined"
    return "before" if t < t10 else ("during" if t <= t99 else "after")


# ============================================================ implementation checks

def check_synthetic(F):
    """C2: synthetic logits with a known S part (§4)."""
    rng = np.random.default_rng(0)
    a, b, _, s = grid_inputs()
    c = np.arange(P)[None, :]
    S = [3, 11, 40]
    alpha = {3: 2.0, 11: -1.3, 40: 0.7}
    A = rng.normal(size=(1, P)) + sum(alpha[k] * np.cos(2 * np.pi * k * (s[:, None] - c) / P) for k in S)
    D = 0.9 * np.cos(2 * np.pi * 7 * (s[:, None] - c) / P)           # a frequency outside S
    f = rng.normal(size=P)[a][:, None] + rng.normal(size=(P, P))[b]   # f(a) + g(b, c)
    f = f - f.mean(0, keepdims=True)                                  # zero mean per column: orthogonal to u
    L = A + D + f
    ctx = Ctx(F, np.arange(P * P), np.arange(P * P), s)
    R = F[:, [0] + fcols(S)] @ (F[:, [0] + fcols(S)].T @ L)
    X = L - proj_S(L, ctx, S)
    S_part = A - A.mean(0, keepdims=True)
    e1 = float(np.abs(R - A).max())
    e2 = float(np.abs(X - (L - S_part)).max())
    return e1 < 1e-9 and e2 < 1e-9, f"restricted err={e1:.2e}, excluded err={e2:.2e}"


# ============================================================ per seed

def load(outputs, tag):
    run = json.loads((outputs / tag / "run.json").read_text())
    cfg = RunConfig(**run["config"])
    ckpts = torch.load(outputs / tag / "checkpoints.pt")
    return run, cfg, ckpts


def model_logits(model, sd, x):
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        return model(x)


def mlp_split(model, x):
    """(logits, mlp-path logits) for the 1-layer model, recomputed by hand (§4 secondary)."""
    blk = model.blocks[0]
    with torch.no_grad():
        h = model.W_E[x] + model.W_pos[: x.shape[1]]
        B, T, D = h.shape
        split = lambda t, n: t.view(B, n, blk.n_heads, blk.d_head).transpose(1, 2)
        q = split(h[:, T - 1:] @ blk.W_Q, 1)
        k, v = split(h @ blk.W_K, T), split(h @ blk.W_V, T)
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(blk.d_head)).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, 1, D)
        r = h[:, T - 1:] + z @ blk.W_O
        mlp = TF.relu(r @ blk.W_in + blk.b_in) @ blk.W_out + blk.b_out
        return ((r + mlp)[:, -1] @ model.W_U).double().numpy(), (mlp[:, -1] @ model.W_U).double().numpy()


def key_set(ckpts, end):
    e = we_spectrum(ckpts[end]["W_E"][:P].double().numpy())
    return sorted(int(k) for k in (np.argsort(-e) + 1)[:TOP]), e


def analyse_seed(job):
    seed, outputs, tag, all_keys = job
    torch.set_num_threads(1)
    run, cfg, ckpts = load(Path(outputs), tag)
    steps = sorted(ckpts)
    assert steps == run["steps"]
    end = steps[-1]
    assert end == cfg.steps
    F = input_basis()
    a, b, xg, y = grid_inputs()
    tr, te = split_idx(cfg)
    ctx = Ctx(F, tr, te, y)
    model = Transformer(cfg)
    rows_pred, rows_final, rows_curve = [], [], []

    def pred(check, ok, detail):
        rows_pred.append({"seed": seed, "check": check, "pass": bool(ok), "detail": detail})

    # ---- implementation checks (§4)
    trx, try_, tex, tey = make_dataset(cfg)
    c4 = torch.equal(xg[tr], trx) and torch.equal(xg[te], tex) and np.array_equal(y[tr], try_.numpy()) \
        and np.array_equal(y[te], tey.numpy())
    pred("C4", c4, "grid rows + permutation reproduce make_dataset train/test tensors exactly")
    lg_tr = model_logits(model, ckpts[end], trx)
    lg_te = model_logits(model, ckpts[end], tex)
    l_tr = TF.cross_entropy(lg_tr.to(torch.float64), try_).item()
    l_te = TF.cross_entropy(lg_te.to(torch.float64), tey).item()
    rel_tr = abs(l_tr - run["train_loss"][-1]) / run["train_loss"][-1]
    rel_te = abs(l_te - run["test_loss"][-1]) / run["test_loss"][-1]
    pred("C1b", rel_tr < 1e-6 and rel_te < 1e-6, f"rel diff train={rel_tr:.2e} test={rel_te:.2e}")
    L_end = model_logits(model, ckpts[end], xg).double().numpy()
    c1c = float(np.abs(L_end[tr] - lg_tr.double().numpy()).max())
    pred("C1c", c1c < 1e-4, f"max |grid - batch| train logits = {c1c:.2e}")
    X0 = L_end - proj_S(L_end, ctx, [])
    pred("C1", np.array_equal(X0, L_end), "excluded with S = empty equals L")
    orth = float(np.abs(F.T @ F - np.eye(F.shape[1])).max())
    pred("C3", orth < 1e-10, f"max|F^T F - I| = {orth:.2e}")
    ok2, d2 = check_synthetic(F)
    pred("C2", ok2, d2)

    # ---- key sets
    key = all_keys[seed]
    key_chk, e = key_set(ckpts, end)
    assert key_chk == key
    le = logit_spectrum(L_end, F)
    logit_rank = [int(k) for k in np.argsort(-le) + 1]
    logit_key = sorted(logit_rank[:TOP])
    logit_6_10 = sorted([k for k in logit_rank if k not in key][:5])
    keyrow = {"seed": seed, "key_freqs_WE": ";".join(map(str, key)),
              "WE_key_energy_final": f"{e[np.array(key) - 1].sum() / e.sum():.4f}",
              "logit_top5": ";".join(map(str, logit_key)),
              "logit_top5_share_of_ab_energy": f"{le[np.array(logit_key) - 1].sum() / le.sum():.4f}",
              "WE_key_share_of_logit_ab_energy": f"{le[np.array(key) - 1].sum() / le.sum():.4f}",
              "overlap_WE_vs_logit": len(set(key) & set(logit_key)),
              "logit_next5_nonkey": ";".join(map(str, logit_6_10)),
              "m3_key_freqs": ";".join(map(str, M3_KEYS.get(seed, []))),
              "overlap_with_m3": len(set(key) & set(M3_KEYS[seed])) if seed in M3_KEYS else ""}

    rng = np.random.default_rng(1000 + seed)
    others = [k for k in range(1, NF + 1) if k not in key]
    rsets = [sorted(int(k) for k in rng.choice(others, TOP, replace=False)) for _ in range(N_RANDOM)]

    # ---- final step (§4 P0-P2 + secondary)
    def frow(cond, freqs, m):
        rows_final.append({"seed": seed, "condition": cond, "freqs": ";".join(map(str, freqs)),
                           **{k: f"{v:.6g}" for k, v in m.items()}})
        return m

    own = {"own_test_acc": acc(L_end[te], y[te]), "own_test_ce": ce(L_end[te], y[te]),
           "own_test_cecal": ce_cal(L_end[te], y[te]), "own_train_acc": acc(L_end[tr], y[tr])}
    frow("model_itself", [], own)
    m_all = frow("all48", range(1, NF + 1), m_restricted_test(L_end, ctx, list(range(1, NF + 1))))
    pred("P0", m_all["restricted_test_acc"] >= 0.99, f"all-48 test-fit restricted test acc = {m_all['restricted_test_acc']:.4f}")
    mk = frow("key", key, all_measures(L_end, ctx, key))
    mr = [frow(f"random{i:03d}", rs, all_measures(L_end, ctx, rs)) for i, rs in enumerate(rsets)]
    frow("logit_rank6_10", logit_6_10, all_measures(L_end, ctx, logit_6_10))
    for s2, ks2 in all_keys.items():
        if s2 != seed:
            frow(f"seed{s2}_keys", ks2, all_measures(L_end, ctx, ks2))
    for k in key:
        frow(f"key_minus_{k}", [j for j in key if j != k], all_measures(L_end, ctx, [j for j in key if j != k]))
    top4 = sorted(int(k) for k in (np.argsort(-e) + 1)[:4])
    frow("key_top4", top4, all_measures(L_end, ctx, top4))
    ml = frow("logit_key", logit_key, all_measures(L_end, ctx, logit_key))

    def p1(m, rand, label):
        r_acc = np.array([x["excluded_train_acc"] for x in rand])
        r_cal = np.array([x["excluded_train_cecal"] for x in rand])
        n99 = int((r_acc >= 0.99).sum())
        ok = m["excluded_train_acc"] <= 0.10 and m["excluded_train_cecal"] >= 0.9 * LN_P and n99 >= 95
        return ok, (f"{label} excl train acc={m['excluded_train_acc']:.4f} (<=0.10), CE*={m['excluded_train_cecal']:.4f} "
                    f"(>= {0.9 * LN_P:.3f}), raw CE={m['excluded_train_ce']:.4g}; random excl acc>=99% in {n99}/100; "
                    f"key CE* > all random CE*: {bool((m['excluded_train_cecal'] > r_cal).all())} (random max {r_cal.max():.4f})")

    def p2(m, rand, label):
        r_acc = np.array([x["restricted_test_acc"] for x in rand])
        r_cal = np.array([x["restricted_test_cecal"] for x in rand])
        n5 = int((r_acc <= 0.05).sum())
        ok = m["restricted_test_acc"] >= 0.99 and n5 >= 95
        return ok, (f"{label} test-fit restr test acc={m['restricted_test_acc']:.4f} (>=0.99), CE*={m['restricted_test_cecal']:.4f}; "
                    f"random restr acc<=5% in {n5}/100 (max {r_acc.max():.4f}); key CE* < all random: "
                    f"{bool((m['restricted_test_cecal'] < r_cal).all())} (random min {r_cal.min():.4f}); "
                    f"leaking full-grid restr test acc={m['restrictedgrid_test_acc']:.4f}")

    pred("P1", *p1(mk, mr, "key"))
    pred("P2", *p2(mk, mr, "key"))
    pred("P1-secondary-logitkey", *p1(ml, mr, "logit-key"))
    pred("P2-secondary-logitkey", *p2(ml, mr, "logit-key"))

    model.load_state_dict(ckpts[end])
    Lchk, Lmlp = mlp_split(model, xg)
    assert np.abs(Lchk - L_end).max() < 1e-4
    PSall = proj_S(L_end, ctx, key)
    nonmlp = float(np.linalg.norm(proj_S(L_end - Lmlp, ctx, key)) / np.linalg.norm(PSall))
    pred("secondary-nonMLP-share-in-S", True, f"||P_S(L - L_mlp)|| / ||P_S L|| = {nonmlp:.4f}")

    # ---- over training
    tr_acc, te_acc = run["train_acc"], run["test_acc"]
    series = {n: [] for n in ["key"] + [f"r{i}" for i in range(N_TRACK)]}
    own_cal, emb = [], []
    for i, st in enumerate(steps):
        L = model_logits(model, ckpts[st], xg).double().numpy()
        row = {"seed": seed, "step": st, "train_acc": f"{tr_acc[i]:.6g}", "test_acc": f"{te_acc[i]:.6g}"}
        own_cal.append(ce_cal(L[te], y[te]))
        row["own_test_cecal"] = f"{own_cal[-1]:.6g}"
        ew = we_spectrum(ckpts[st]["W_E"][:P].double().numpy())
        emb.append(float(ew[np.array(key) - 1].sum() / ew.sum()))
        row["emb_key_energy"] = f"{emb[-1]:.6g}"
        for name, fr in [("key", key)] + [(f"r{j}", rsets[j]) for j in range(N_TRACK)]:
            m = all_measures(L, ctx, fr)
            series[name].append(m)
            for k_, v in m.items():
                row[f"{name}_{k_}"] = f"{v:.6g}"
        rows_curve.append(row)

    t_mem = first_step(steps, tr_acc, lambda v: v >= 0.99)
    t10, t50, t95, t99 = (first_step(steps, te_acc, lambda v, q=q: v >= q) for q in (0.10, 0.50, 0.95, 0.99))
    ser = lambda name, k: [m[k] for m in series[name]]

    # P3
    kacc = ser("key", "restricted_test_acc")
    rmax = np.max([ser(f"r{j}", "restricted_test_acc") for j in range(N_TRACK)], axis=0)
    tR50 = first_step(steps, kacc, lambda v: v >= 0.5)
    tRm = first_step(steps, list(np.array(kacc) - rmax), lambda v: v >= 0.30)
    pred("P3", tR50 is not None and tRm is not None and tR50 < t50 and tRm < t50,
         f"key restr acc>=50% at {tR50}; key-minus-max-random>=0.30 at {tRm}; t50={t50}")
    # P4
    tphi = first_step(steps, ser("key", "margin_share"), lambda v: v >= 0.5, start=t_mem)
    pred("P4", tphi is not None and tphi < t50, f"phi_key>=0.5 at {tphi}; t50={t50}")
    tX50 = first_step(steps, ser("key", "excluded_train_acc"), lambda v: v < 0.5, start=t_mem)
    pred("P4-report", True, f"excluded train acc <50% first at {tX50}; t50={t50}; t95={t95}")

    ev = {"seed": seed, "t_mem": t_mem, "t10": t10, "t50": t50, "t95": t95, "t99": t99,
          "t_restricted_acc50": tR50, "t_restricted_margin030": tRm, "t_phi050": tphi, "t_excluded_acc_below50": tX50}
    measures_p5 = [("m1_restricted_cecal", ser("key", "restricted_test_cecal"), -1, "ce", True),
                   ("m2_excluded_cecal", ser("key", "excluded_train_cecal"), +1, "ce", True),
                   ("m3_emb_key_energy", emb, +1, "energy", False),
                   ("ctrl_own_test_cecal", own_cal, -1, "ce", False)]
    for j in range(N_TRACK):
        measures_p5 += [(f"ctrl_r{j}_restricted_cecal", ser(f"r{j}", "restricted_test_cecal"), -1, "ce", False),
                        (f"ctrl_r{j}_excluded_cecal", ser(f"r{j}", "excluded_train_cecal"), +1, "ce", False)]
    end20k = max(s for s in steps if s <= 20000)
    for name, m, direction, kind, is_pred in measures_p5:
        for endpoint, suffix in ((end, ""), (end20k, "@20k")):
            r = events(steps, m, t_mem, endpoint, direction, kind)
            ev[f"{name}{suffix}_defined"] = r["defined"]
            ev[f"{name}{suffix}_delta"] = f"{r['delta']:.4g}"
            for evname, want in (("onset", "before"), ("steepest", "during"), ("completion", "after")):
                t = r[evname]
                ph = phase(t, t10, t99)
                ev[f"{name}{suffix}_{evname}"] = "" if t is None else f"{t:.0f}"
                ev[f"{name}{suffix}_{evname}_phase"] = ph
                if t is not None:
                    ev[f"{name}{suffix}_{evname}_ln_minus_ln_t10"] = f"{math.log(t) - math.log(t10):.3f}"
                    ev[f"{name}{suffix}_{evname}_ln_minus_ln_t50"] = f"{math.log(t) - math.log(t50):.3f}"
                    ev[f"{name}{suffix}_{evname}_ln_minus_ln_t99"] = f"{math.log(t) - math.log(t99):.3f}"
                if suffix == "" and (is_pred or name == "m3_emb_key_energy"):
                    pred(f"P5:{name}:{evname}", ph == want,
                         f"{evname}={'' if t is None else round(t)} phase={ph} (want {want}); t10={t10} t99={t99}")

    # P6
    after = [i for i, s in enumerate(steps) if s >= t_mem]
    details, ok6 = [], True
    for j in range(N_TRACK):
        fr = np.mean([series[f"r{j}"][i]["restricted_test_acc"] <= 0.05 for i in after])
        fx = np.mean([abs(series[f"r{j}"][i]["excluded_train_acc"] - tr_acc[i]) <= 0.01 for i in after])
        ok6 &= fr >= 0.95 and fx >= 0.95
        details.append(f"r{j}: restr<=5% {fr:.3f}, excl within 1pp {fx:.3f}")
    pred("P6", ok6, "; ".join(details))

    repro = None
    if seed in README_GROK:
        ref = README_GROK[seed]
        repro = {"seed": seed, "readme_m3": ref, "linux_rerun": t95, "diff_steps": t95 - ref,
                 "diff_pct": f"{100 * (t95 - ref) / ref:.2f}"}
    print(f"seed {seed} done: key={key} t95={t95}", flush=True)
    return {"seed": seed, "pred": rows_pred, "final": rows_final, "curve": rows_curve,
            "events": ev, "keyrow": keyrow, "repro": repro}


# ============================================================ main

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="circuit__seed{seed}")
    ap.add_argument("--out", default=str(ROOT / "notes" / "circuit"))
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    all_keys = {}
    for s in args.seeds:
        ck = torch.load(Path(args.outputs) / args.tag.format(seed=s) / "checkpoints.pt")
        all_keys[s] = key_set(ck, max(ck))[0]
        del ck
    jobs = [(s, args.outputs, args.tag.format(seed=s), all_keys) for s in args.seeds]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            res = list(ex.map(analyse_seed, jobs))
    else:
        res = [analyse_seed(j) for j in jobs]

    reps = [r["repro"] for r in res if r["repro"]]
    if reps:
        write_csv(out / "circuit_repro.csv", reps)
    write_csv(out / "circuit_keyfreqs.csv", [r["keyrow"] for r in res])
    write_csv(out / "circuit_final.csv", [x for r in res for x in r["final"]])
    write_csv(out / "circuit_over_training.csv", [x for r in res for x in r["curve"]])
    write_csv(out / "circuit_events.csv", [r["events"] for r in res])
    preds = [x for r in res for x in r["pred"]]
    write_csv(out / "circuit_predictions.csv", preds)

    n = len(args.seeds)
    print("\ncheck                                         passes / seeds")
    for check in dict.fromkeys(r["check"] for r in preds):
        k = sum(r["pass"] for r in preds if r["check"] == check)
        print(f"{check:45s} {k}/{n}")
    plot(out, [x for r in res for x in r["curve"]], [r["events"] for r in res], args.seeds)


def plot(out, curve, evrows, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(seeds), 2, figsize=(11, 2.4 * len(seeds)), sharex=True, squeeze=False)
    for (ax, ax2), seed in zip(axes, seeds):
        rows = [r for r in curve if r["seed"] == seed]
        st = np.array([r["step"] for r in rows], float)
        f = lambda k: np.array([float(r[k]) for r in rows])
        ax.plot(st, f("test_acc"), color="#eb6834", lw=1.6, label="test acc (model)")
        ax.plot(st, f("key_restricted_test_acc"), color="#2a78d6", lw=1.4, label="restricted test acc, key (test-fit)")
        ax.plot(st, f("key_excluded_train_acc"), color="#7a4ec2", lw=1.4, label="excluded train acc, key")
        ax.plot(st, f("key_margin_share"), color="#2e9e6b", lw=1.2, label="key margin share φ (train)")
        ax.plot(st, f("emb_key_energy"), color="#52514e", lw=1.0, ls="--", label="W_E key energy")
        for j in range(N_TRACK):
            ax.plot(st, f(f"r{j}_restricted_test_acc"), color="#2a78d6", lw=0.5, alpha=0.35)
            ax.plot(st, f(f"r{j}_excluded_train_acc"), color="#7a4ec2", lw=0.5, alpha=0.35)
        ax2.plot(st, f("own_test_cecal"), color="#eb6834", lw=1.4, label="model test CE*")
        ax2.plot(st, f("key_restricted_test_cecal"), color="#2a78d6", lw=1.4, label="restricted test CE*, key")
        ax2.plot(st, f("key_excluded_train_cecal"), color="#7a4ec2", lw=1.4, label="excluded train CE*, key")
        for j in range(N_TRACK):
            ax2.plot(st, f(f"r{j}_restricted_test_cecal"), color="#2a78d6", lw=0.5, alpha=0.35)
            ax2.plot(st, f(f"r{j}_excluded_train_cecal"), color="#7a4ec2", lw=0.5, alpha=0.35)
        ev = next(e for e in evrows if e["seed"] == seed)
        for a_ in (ax, ax2):
            for t in (ev["t10"], ev["t99"]):
                a_.axvline(t, color="#b8b6ae", lw=0.8)
            a_.set_xscale("log")
            a_.grid(True, color="#e8e6df")
        ax.set_ylim(-0.05, 1.05)
        ax2.set_ylim(-0.1, LN_P + 0.2)
        ax.set_ylabel(f"seed {seed}")
    axes[0][0].legend(fontsize=6.5, loc="center left")
    axes[0][1].legend(fontsize=6.5, loc="center left")
    axes[0][0].set_title("accuracies / shares", fontsize=9)
    axes[0][1].set_title("calibrated CE* (nats; ln 97 = no information)", fontsize=9)
    for a_ in axes[-1]:
        a_.set_xlabel("step (log); grey: test acc 10% and 99%; faint: 5 random sets")
    fig.tight_layout()
    fig.savefig(out / "circuit_over_training.png", dpi=130)
    print(f"wrote {out / 'circuit_over_training.png'}")


if __name__ == "__main__":
    main()
