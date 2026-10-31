#!/usr/bin/env bash
# Reproduce everything end to end: train all runs, project, export, validate, plot.
# Usage: ./run_all.sh            (uses ./.venv/bin/python if present, else python3)
#        JOBS=4 ./run_all.sh     (train runs in parallel; identical results, ~10% faster on an M3)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
JOBS=${JOBS:-1}
RUNS="baseline no_decay more_data less_data"

printf '%s\n' $RUNS | xargs -P "$JOBS" -I{} "$PY" train/train.py --run {}
"$PY" train/project.py $RUNS
"$PY" train/export.py
"$PY" train/validate.py
"$PY" train/plot.py curve outputs/baseline notes/plots/phase0_baseline.png
"$PY" train/plot.py sweep notes/plots/phase1_sweep.png $(for r in $RUNS; do echo outputs/$r; done)
"$PY" train/plot.py summary $(for r in $RUNS; do echo outputs/$r; done)
