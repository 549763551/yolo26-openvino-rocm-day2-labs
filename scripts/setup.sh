#!/usr/bin/env bash
# =============================================================================
# YOLO26 Day 02 — One-click setup
# Target: Linux + AMD CPU + Radeon PRO W7900 (gfx1100), Python 3.12
# Everything is stored inside the repository (persistent on /workspace).
# Idempotent: re-running skips completed steps.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"
pipv() { "$VENV/bin/python" -m pip "$@"; }
WEIGHTS_DIR="$REPO_ROOT/weights"
DATA_DIR="$REPO_ROOT/data"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$WEIGHTS_DIR" "$DATA_DIR" "$LOG_DIR"

say()  { printf '\033[1;36m[setup %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m  [OK]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m  [FAIL]\033[0m %s\n' "$*"; exit 1; }

say "Step 1/10 — Hardware gate: Linux + AMD CPU + gfx1100 (W7900)"
[[ "$(uname -s)" == "Linux" ]] || fail "Not Linux."
VENDOR="$(grep -m1 vendor_id /proc/cpuinfo | awk '{print $3}')"
[[ "$VENDOR" == "AuthenticAMD" ]] || fail "CPU vendor is '$VENDOR', expected AuthenticAMD. CPU labs must run on AMD CPU."
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
ok "AMD CPU: $CPU_MODEL"
GFX=""
if command -v rocminfo >/dev/null 2>&1; then
  GFX="$(rocminfo 2>/dev/null | grep -oE 'gfx11[0-9]{2}' | sort -u | tr '\n' ' ' | sed 's/ $//')"
fi
if [[ "$GFX" == *gfx1100* ]]; then
  ok "AMD GPU with gfx1100 detected (Radeon PRO W7900 class): $GFX"
else
  fail "No gfx1100 GPU found via rocminfo (got: '${GFX:-none}'). GPU labs require the W7900."
fi

say "Step 2/10 — Python 3.12 virtual environment at .venv/"
if [[ ! -x "$PY" ]]; then
  command -v python3.12 >/dev/null 2>&1 || fail "python3.12 not found."
  python3.12 -m venv "$VENV"
  ok "Created $VENV"
else
  ok "Reusing existing .venv"
fi
"$PY" --version | grep -q "3.12" || fail "Venv is not Python 3.12"

say "Step 3/10 — AMD ROCm PyTorch (user-specified wheels, installed verbatim)"
# Reinstall check: exact versions already present → skip.
if "$PY" - <<'EOF'
import sys
try:
    import torch, torchvision, torchaudio
    assert torch.__version__ == "2.12.0+rocm7.14.1"
    assert torchvision.__version__ == "0.27.0+rocm7.14.1"
    assert torchaudio.__version__ == "2.11.0+rocm7.14.1"
    sys.exit(0)
except Exception:
    sys.exit(1)
EOF
then
  ok "ROCm PyTorch already at pinned versions."
else
  # NOTE: this is the exact user-specified install command. Do not modify.
  pipv install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
      "torch[device-gfx1100]==2.12.0+rocm7.14.1" \
      "torchvision[device-gfx1100]==0.27.0+rocm7.14.1" \
      "torchaudio==2.11.0+rocm7.14.1" 2>&1 | tee "$LOG_DIR/pip_rocm_torch.log" | tail -5
  ok "ROCm PyTorch installed."
fi

say "Step 4/10 — Course dependencies (constrained so AMD wheels are never replaced)"
pipv install -c "$REPO_ROOT/constraints-rocm.txt" \
    --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "ultralytics>=8.3.0" "openvino>=2024.4" "nncf>=2.13" \
    "matplotlib>=3.8" "pandas>=2.1" "numpy>=1.26" "pillow>=10.0" "tqdm>=4.66" \
    2>&1 | tee "$LOG_DIR/pip_deps.log" | tail -3
ok "Course dependencies installed."

say "Step 5/10 — Jupyter stack + kernel registration"
pipv install -c "$REPO_ROOT/constraints-rocm.txt" \
    --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "jupyterlab>=4.2" "ipykernel>=6.29" "ipywidgets>=8.1" 2>&1 | tail -2
if ! jupyter kernelspec list 2>/dev/null | grep -q "yolo26-day2"; then
  "$VENV/bin/python" -m ipykernel install --user --name yolo26-day2 \
      --display-name "Python 3.12 (YOLO26 Day 02 · ROCm)" >/dev/null
fi
ok "Jupyter kernel 'yolo26-day2' ready."

say "Step 6/10 — Official YOLO26l weights (Ultralytics source, SHA256 recorded)"
export PYTHONPATH="$REPO_ROOT"
"$PY" - <<'EOF'
import hashlib, json, sys
from pathlib import Path
from ultralytics import YOLO

wdir = Path("weights"); wdir.mkdir(exist_ok=True)
pt = wdir / "yolo26l.pt"
if not pt.exists():
    print("  downloading official yolo26l.pt (Ultralytics release asset) ...")
    import os, shutil, subprocess
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
        "https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26l.pt",  # official Ultralytics HF org
    ]
    ok = False
    for url in urls:
        r = subprocess.run(["curl", "-L", "-sS", "--retry", "3", "--retry-all-errors",
                            "--connect-timeout", "30", "--max-time", "600",
                            "-o", str(pt) + ".dl", url])
        if r.returncode == 0 and (pt.with_suffix(".pt.dl").exists()
                                  and pt.with_suffix(".pt.dl").stat().st_size > 10_000_000):
            pt.with_suffix(".pt.dl").rename(pt)
            ok = True
            break
        pt.with_suffix(".pt.dl").unlink(missing_ok=True)
    if not ok:
        YOLO("yolo26l.pt")  # last resort: ultralytics downloader
        import glob
        for cand in glob.glob("yolo26l.pt"):
            shutil.move(cand, pt)
    assert pt.exists(), "download failed"
sha = hashlib.sha256(pt.read_bytes()).hexdigest()
model = YOLO(str(pt))
names = model.names
n = len(names)
task = getattr(model.model, "task", None)
assert n == 80, f"expected 80 COCO classes, got {n}"
print(f"  weights : {pt}  ({pt.stat().st_size/1e6:.1f} MB)")
print(f"  sha256  : {sha}")
print(f"  classes : {n}  task: {task}")
meta = {"file": str(pt), "size_bytes": pt.stat().st_size, "sha256": sha,
        "num_classes": n, "task": str(task),
        "source": "official Ultralytics YOLO('yolo26l.pt') download",
        "ultralytics_version": __import__("ultralytics").__version__}
(wdir / "model_identity.json").write_text(json.dumps(meta, indent=2))
assert task == "detect", f"expected detection task, got {task}"
print("  identity recorded -> weights/model_identity.json")
EOF
ok "Official YOLO26l verified (80-class COCO detection)."

say "Step 7/10 — COCO educational subsets (verify REAL files, repair if incomplete)"
# v2.0.1 fix (defect B-01): the committed manifest alone is NOT proof that the
# images exist. Always verify real image counts, labels, yaml, calib list and
# per-file sha256; run the incremental repair (prepare_data.py) when anything
# is missing or corrupt; re-verify and FAIL loudly if repair did not fix it.
if "$PY" scripts/verify_data.py; then
  ok "Data subsets verified (images + labels + yaml + calib + sha256)."
else
  say "Data incomplete/corrupt — incremental download/repair via prepare_data.py"
  "$PY" scripts/prepare_data.py
  "$PY" scripts/verify_data.py || fail "Data repair did not produce a valid dataset."
  ok "Data subsets repaired and re-verified."
fi

say "Step 8/10 — Environment validation (versions + real compute on AMD CPU and W7900)"
"$PY" course/setup_checks.py 2>&1 | tee "$LOG_DIR/setup_checks.log"

say "Step 9/10 — Quick self-test of course tooling"
"$PY" -m pytest -q tests/test_units.py 2>&1 | tail -2

say "10/10 — Setup complete."
cat <<EOF

  Repository : $REPO_ROOT
  CPU        : $CPU_MODEL
  GPU        : gfx1100 (Radeon PRO W7900)
  Weights    : weights/yolo26l.pt (official, 80-class COCO detection)

  Next steps / 下一步:
    bash scripts/doctor.sh                      # full environment report
    bash scripts/run_all.sh --profile quick     # fast in-class results (~10-20 min)
    bash scripts/run_all.sh --profile full      # formal benchmark (longer)
    source .venv/bin/activate && jupyter lab    # open the six notebooks

  To restore this environment later (Radeon Cloud):
    bash scripts/restore_env.sh
EOF
