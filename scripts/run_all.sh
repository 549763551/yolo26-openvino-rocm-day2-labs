#!/usr/bin/env bash
# =============================================================================
# run_all.sh — run the six real benchmarks headlessly.
#   --profile quick : classroom mode (fewer repetitions, same real model)
#   --profile full  : formal citable benchmark
#   --only labA,labB... : subset of experiments
# Charts and results/latest.json are regenerated from real measured data.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"

PROFILE="quick"
ONLY="environment,labA,labB,labC,labD,labE"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --only)    ONLY="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

[[ -x "$PY" ]] || { echo "no .venv — run scripts/setup.sh first"; exit 1; }
export PYTHONPATH="$REPO_ROOT"

echo "=============================================================="
echo " YOLO26 Day 02 benchmarks · profile=$PROFILE"
echo " model: official yolo26l.pt (never swapped for a smaller one)"
echo "=============================================================="
"$PY" scripts/run_experiments.py --profile "$PROFILE" --only "$ONLY"
echo
echo "Next: open notebooks/*.ipynb in Jupyter Lab, or see assets/slides/."
