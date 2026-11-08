#!/usr/bin/env bash
# =============================================================================
# restore_env.sh — restore the course environment on a fresh Radeon Cloud
# instance where /workspace persists but the VM was recycled.
#
# What persists under /workspace (repo dir): .venv, weights/, data/, results/.
# If the repo directory exists with .venv intact, this script only re-runs the
# health gates. Otherwise it re-runs setup.sh (which is idempotent).
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[restore] repo: $REPO_ROOT"
if [[ -x "$REPO_ROOT/.venv/bin/python" && -f "$REPO_ROOT/weights/yolo26l.pt" && -f "$REPO_ROOT/data/manifest.json" ]]; then
  echo "[restore] venv + weights + data present — verifying health gates"
  bash scripts/doctor.sh
  echo "[restore] environment OK. Nothing to do."
else
  echo "[restore] incomplete environment — running full setup (idempotent)"
  bash scripts/setup.sh
fi
