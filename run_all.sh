#!/usr/bin/env bash
# Reproduce everything end to end: train every run x seed, project, export,
# validate, plot.
# Usage: ./run_all.sh            (uses ./.venv/bin/python if present, else python3)
#        JOBS=4 ./run_all.sh     (train runs in parallel; identical results, ~10% faster on an M3)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
JOBS=${JOBS:-1}

# Run ids and seeds come from train/config.py (SWEEP, SEEDS).
RUNS=$("$PY" train/config.py runs)
SEEDS=$("$PY" train/config.py seeds)

for run in $RUNS; do for seed in $SEEDS; do echo "$run $seed"; done; done \
  | xargs -P "$JOBS" -L 1 sh -c '"$0" train/train.py --run "$1" --seed "$2"' "$PY"
"$PY" train/project.py $RUNS
"$PY" train/export.py
"$PY" train/validate.py
"$PY" train/plot.py curve outputs/baseline notes/plots/phase0_baseline.png
"$PY" train/plot.py sweep notes/plots/phase1_sweep.png
"$PY" train/plot.py summary $("$PY" train/config.py outputs)
