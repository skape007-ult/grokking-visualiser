#!/usr/bin/env bash
# Reproduce everything end to end: train all runs, project, export, validate, plot.
# Usage: ./run_all.sh            (uses ./.venv/bin/python if present, else python3)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
RUNS="baseline no_decay more_data less_data"

for run in $RUNS; do
  "$PY" train/train.py --run "$run"
done
"$PY" train/project.py $RUNS
"$PY" train/export.py
"$PY" train/validate.py
"$PY" train/plot.py curve outputs/baseline notes/plots/phase0_baseline.png
"$PY" train/plot.py sweep notes/plots/phase1_sweep.png $(for r in $RUNS; do echo outputs/$r; done)
"$PY" train/plot.py summary $(for r in $RUNS; do echo outputs/$r; done)
