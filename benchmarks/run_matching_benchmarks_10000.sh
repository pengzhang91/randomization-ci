#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PY="${PY:-python}"

"$PY" "$SCRIPT_DIR/benchmark_matching_exact_ci.py" \
  --pattern balanced \
  --its 10000 \
  --n-list 20 50 100 200 500 1000 2000 4000 10000 \
  --output matching_runtime_balanced_its10000.csv

"$PY" "$SCRIPT_DIR/benchmark_matching_exact_ci.py" \
  --pattern asymmetric \
  --its 10000 \
  --n-list 20 50 100 200 500 1000 2000 4000 10000 \
  --output matching_runtime_asymmetric_its10000.csv
