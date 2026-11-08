#!/usr/bin/env bash
# =============================================================================
# validate_notebooks.sh — notebook gate (v2.0.1)
#
#   bash scripts/validate_notebooks.sh             # structural check (CI-safe)
#   bash scripts/validate_notebooks.sh --execute   # REAL execution of all six
#
# The structural mode checks names/template/bilingual/student-knobs without a
# GPU (used in GitHub Actions). The --execute mode runs every code cell in the
# project environment via nbclient and verifies expected outputs — this is the
# release gate; a notebook that does not RUN does not PASS.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

if [[ "${1:-}" == "--execute" ]]; then
  echo "== Notebook EXECUTION gate (real runs on this machine) =="
  exec "$PY" scripts/execute_notebooks.py "${@:2}"
else
  echo "== Notebook STRUCTURE gate (bilingual template, student knobs) =="
  exec "$PY" tests/validate_notebooks.py
fi
