"""Run configuration.

Every training run is fully described by a RunConfig. Nothing that affects
training is hard-coded inside functions: change a field here (or pass
`--set key=value` to train.py), never a constant buried in code. The full config
is written into every output file next to the results it produced.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class RunConfig:
    id: str = "baseline"

    # Task: (a + b) mod p, inputs are the 4 tokens [a, op, b, =].
    p: int = 97
    operation: str = "(a + b) mod p"
    train_frac: float = 0.3
    seed: int = 0

    # Model: 1-layer transformer in the style of Nanda et al. 2023
    # (no LayerNorm, learned positional embeddings, untied embed/unembed).
    n_layers: int = 1
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 4

    # Optimisation: full-batch AdamW with decoupled weight decay.
    steps: int = 25_000
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 10
    # Cross-entropy is computed in this dtype (float64 helps late-training
    # stability). MPS has no float64, so train.py falls back to float32 there
    # and records the dtype actually used.
    loss_dtype: str = "float64"

    # Recording.
    n_record: int = 200      # log-spaced steps with metrics (the base schedule)
    n_snapshots: int = 40    # of those, log-spaced steps with embedding snapshots
    # Adaptive extras, so the transition isn't under-sampled. When test accuracy
    # is changing faster than `dense_trigger_rate` per e-fold of steps, also
    # record every `dense_ratio` (2%) of steps; and snapshot the embeddings
    # whenever test accuracy has moved `snapshot_delta` since the last snapshot.
    dense_trigger_rate: float = 0.2
    dense_ratio: float = 0.02
    snapshot_delta: float = 0.05

    # "auto" = CUDA -> MPS -> CPU. Pinned to CPU by default: on the Apple M3 used
    # here CPU and MPS run this model at the same speed (~13 ms/step), and CPU
    # supports the float64 loss and gives bit-for-bit reproducible runs.
    device: str = "cpu"

    @property
    def output_name(self) -> str:
        """Directory under outputs/: the run id, plus the seed if not the default."""
        return self.id if self.seed == BASE_SEED else f"{self.id}__seed{self.seed}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def with_overrides(self, **kwargs) -> "RunConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RunSpec:
    """A named run in the sweep: its config plus how the site presents it."""

    config: RunConfig
    label: str
    description: str


BASE_SEED = 0
# Every condition is trained with each of these seeds. The base seed's run is
# the one shown in full on the site (embeddings included); the others are
# drawn as faint accuracy curves to show run-to-run variation.
SEEDS = (0, 1, 2)

BASELINE = RunConfig(id="baseline")

# The sweep. Each variant differs from BASELINE in exactly one field.
# Order here is the order shown on the site.
SWEEP: list[RunSpec] = [
    RunSpec(BASELINE, "Standard training", "Weight decay 1.0, 30% of data"),
    RunSpec(
        BASELINE.with_overrides(id="no_decay", weight_decay=0.0),
        "No weight decay",
        "Weight decay 0, 30% of data",
    ),
    RunSpec(
        BASELINE.with_overrides(id="more_data", train_frac=0.6),
        "More training data",
        "Weight decay 1.0, 60% of data",
    ),
    RunSpec(
        BASELINE.with_overrides(id="less_data", train_frac=0.15),
        "Less training data",
        "Weight decay 1.0, 15% of data",
    ),
]

RUNS: dict[str, RunSpec] = {spec.config.id: spec for spec in SWEEP}
RUN_ORDER: list[str] = [spec.config.id for spec in SWEEP]


def seed_configs(spec: RunSpec) -> list[RunConfig]:
    """The run's config with each seed in SEEDS (base seed first)."""
    return [spec.config.with_overrides(seed=s) for s in SEEDS]


def parse_override(cfg: RunConfig, assignment: str) -> RunConfig:
    """Apply a `key=value` override, casting value to the field's type."""
    key, _, raw = assignment.partition("=")
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    if key not in fields:
        raise KeyError(f"unknown config field {key!r}")
    current = getattr(cfg, key)
    value = type(current)(raw) if not isinstance(current, bool) else raw.lower() in ("1", "true")
    return replace(cfg, **{key: value})


if __name__ == "__main__":
    # Used by run_all.sh: `python train/config.py runs|seeds|outputs`.
    import sys

    what = sys.argv[1]
    if what == "runs":
        print(" ".join(RUN_ORDER))
    elif what == "seeds":
        print(" ".join(map(str, SEEDS)))
    elif what == "outputs":
        print(" ".join(f"outputs/{c.output_name}" for spec in SWEEP for c in seed_configs(spec)))
    else:
        sys.exit(f"unknown: {what}")
