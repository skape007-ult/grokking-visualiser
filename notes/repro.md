# Reproducibility check on Linux CPU — INTERIM (runs still in progress)

**Status at 01:55 America/Toronto, 2026-11-05: no run has finished, so (a) and
(b) are only partly answered.** This file will be updated when the runs finish.

## Setup

- Machine: Linux VM, 4 vCPU Intel Xeon @ 2.10 GHz, kernel 6.18.44, Python 3.13.18.
- `pip install -r requirements.txt` → torch 2.15.0 (the default Linux wheel,
  `2.15.0+cu132`; no GPU, so it runs on CPU), numpy 2.5.4, matplotlib 3.11.2.
- Reference: Apple M3 CPU, Python 3.14 (per README).
- Commands (the documented ones, from the README):
  - `python train/train.py --run baseline` (seed 0)
  - `python train/train.py --run baseline --seed 1`
  - `python train/train.py --run baseline --seed 2`

  These three ran in parallel and started 00:10 EST. The seed-0 rerun,
  `... --run baseline --tag baseline__seed0_rerun`, started 00:21 and overlapped
  with the other three. Code was unchanged.

## Why no run finished

A training step takes about **1.5–3 s** here, against about 13 ms on the M3
(README). The three parallel runs reached only step 1,766 of 25,000 after 85
minutes. That is 4 processes × 4 torch threads on 4 vCPUs. The rerun started
later but moved faster (step 9,675 at the same time), so contention between
the runs is uneven. At this rate a full baseline run takes many hours, which
does not fit the 02:30 deadline. I changed nothing to speed it up (thread
counts, device): either change could itself alter the floating-point results.

## (a) Seed 0 vs seed-0 rerun on this machine

- **`run.json` / `embeddings.npy` comparison: not done yet.** These files are
  written only when a run ends. When both runs finish, compare with
  `np.array_equal` and `max |a-b|` on every metric array and on the
  embeddings.
- **Interim evidence:** the console log prints train/test accuracy and loss at
  29 recorded steps up to step 1,766. All 29 lines are **identical** between
  the two runs (at the printed precision; timing column excluded). This is
  consistent with deterministic training on one machine, but it is not proof
  of bit-identity.

## (b) Grokking step (first recorded step with test acc ≥ 95%)

| seed | Apple M3 (reported) | this Linux CPU VM |
|---|---|---|
| 0 | 4,216 | **not yet; clearly later than 4,216.** Rerun log: test acc 8.1% at step 4,563, 10.8% at 5,785, 20.0% at 7,631, 88.0% at 9,675 |
| 1 | 5,785 | pending (7.1% at step 1,766) |
| 2 | 9,868 | pending (10.9% at step 1,766) |

Seed 0 already differs from the M3 here. At step 4,563 this machine's test
accuracy is 8.1%, while on the M3 the run had passed 95% by step 4,216.

## Why cross-hardware results differ (expected)

`torch.use_deterministic_algorithms(True)` plus a fixed seed makes a run
repeatable **on the same hardware and software stack**. It does not make runs
match across machines:

- **Floating-point operation order.** Float addition is not associative. Matmul
  and reduction kernels split and sum in different orders depending on the
  CPU's SIMD width (x86 AVX2/AVX-512 vs ARM NEON/SME), the blocking strategy
  and the thread count. This changes the last bits of the results.
- **Backend kernels.** On x86 Linux, torch CPU matmuls go through oneDNN/MKL-style
  kernels. On macOS ARM they go through Accelerate/other ARM kernels. These are
  different implementations, and so are libm's `exp`/`log` and the Python
  versions (3.13 vs 3.14).
- **Amplification.** Grokking takes tens of thousands of AdamW steps with strong
  weight decay. In that regime tiny last-bit differences compound, much like
  different seeds do, and the 95% crossing step moves by thousands of steps.
  The README already notes that the step varies 2.3× across seeds.

The README's "deterministic" claim is therefore best read as *same-machine*
reproducibility. No fix was attempted (out of scope).
