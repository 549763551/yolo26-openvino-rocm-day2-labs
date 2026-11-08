#!/usr/bin/env bash
# =============================================================================
# doctor.sh — full environment diagnostic report (read-only, no installs)
# Exits non-zero if any hard gate fails.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"

hr() { printf '\n\033[1;35m==== %s ====\033[0m\n' "$*"; }

hr "1. OS / AMD CPU"
grep -m1 vendor_id /proc/cpuinfo
grep -m1 "model name" /proc/cpuinfo
lscpu | grep -E "^(CPU\(s\)|Socket|Core|Thread|NUMA node\(s\))" | head -6

hr "2. GPU / ROCm kernel"
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E "Marketing Name|gfx|Device Type" | head -8
else
  echo "rocminfo not found"
fi
rocm-smi --showproductname --showuse --showmeminfo vram 2>/dev/null | head -14 || true

hr "3. PyTorch ROCm versions + REAL compute"
if [[ -x "$PY" ]]; then
  "$PY" - <<'EOF'
import torch, torchvision
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("hip         :", torch.version.hip)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    a = torch.randn(1024, 1024, device="cuda"); b = a @ a
    torch.cuda.synchronize()
    print("gpu matmul  :", "OK" if torch.isfinite(b).all().item() else "BAD")
    from torchvision.ops import nms
    keep = nms(torch.tensor([[0,0,10,10],[1,1,11,11],[50,50,60,60]], dtype=torch.float32, device="cuda"),
               torch.tensor([.9,.8,.7], device="cuda"), .5)
    print("gpu nms     :", "OK" if keep.tolist() == [0,2] else "BAD")
assert torch.__version__ == "2.12.0+rocm7.14.1", "torch version drift!"
assert torchvision.__version__ == "0.27.0+rocm7.14.1", "torchvision version drift!"
print("version pins: OK (2.12.0+rocm7.14.1 / 0.27.0+rocm7.14.1)")
EOF
else
  echo "MISSING: .venv — run scripts/setup.sh first"; exit 1
fi

hr "4. OpenVINO / Ultralytics / NNCF"
"$PY" - <<'EOF'
# Import torch FIRST: openvino's AMD GPU plugin loads the system ROCm runtime,
# which must not shadow torch's bundled ROCm userspace libraries.
import torch  # noqa: F401
import openvino as ov
print("openvino    :", ov.__version__, "devices:", ov.Core().available_devices)
import ultralytics; print("ultralytics:", ultralytics.__version__)
import nncf; print("nncf        :", nncf.__version__)
EOF

hr "5. Official YOLO26l identity"
"$PY" - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from course.model import identity_record, OFFICIAL_PT
r = identity_record()
print("file        :", r["file"])
print("size        :", f"{r['size_bytes']/1e6:.1f} MB")
print("sha256      :", r["sha256"])
print("classes     :", r["verified"]["num_classes"], "task:", r["verified"]["task"])
assert r["verified"]["num_classes"] == 80 and r["verified"]["task"] == "detect"
print("identity    : OK — official YOLO26l COCO detection, 80 classes")
EOF

hr "6. Data subsets (REAL file verification, v2.0.1)"
"$PY" scripts/verify_data.py

hr "7. Exports / results"
ls -la weights/ 2>/dev/null | grep -E "yolo26l|int8|openvino" || echo "(no exports yet)"
ls results/raw/*/ 2>/dev/null | head -20 || echo "(no results yet — run scripts/run_all.sh)"
ls assets/slides/ 2>/dev/null || echo "(no charts yet)"

hr "8. Hard gates"
"$PY" - <<'EOF'
import sys; sys.path.insert(0, ".")
from course.setup_checks import full_report, assert_healthy
assert_healthy(full_report())
print("ALL HARD GATES PASS ✓  (AMD CPU + gfx1100 W7900 + ROCm PyTorch + OpenVINO CPU)")
EOF
printf '\nDoctor complete. / 诊断完成。\n'
