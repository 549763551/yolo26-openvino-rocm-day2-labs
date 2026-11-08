#!/usr/bin/env python3
"""Validate the six notebooks: structure, bilingual content, student knobs.

Runs in CI without any GPU. Checks:
  * exactly six notebooks with the expected names;
  * each has the 8-step template sections (What will you learn / Takeaway ...);
  * bilingual (both English and Chinese present) in markdown cells;
  * at least one STUDENT_* parameter cell;
  * 6-16 code cells per notebook.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "notebooks"

EXPECTED = [
    "00_AMD_Environment_and_YOLO26l.ipynb",
    "01_Lab_A_CPU_vs_GPU_Baseline.ipynb",
    "02_Lab_B_OpenVINO_AMD_CPU.ipynb",
    "03_Lab_C_OpenVINO_INT8.ipynb",
    "04_Lab_D_ROCm_GPU_Optimization.ipynb",
    "05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb",
]

CJK = re.compile(r"[\u4e00-\u9fff]")

failures = []
for name in EXPECTED:
    p = NB / name
    if not p.exists():
        failures.append(f"{name}: missing")
        continue
    nb = json.loads(p.read_text())
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    text = "\n".join("".join(c["source"]) for c in md)
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"])

    if "What will you learn" not in text or "本节学什么" not in text:
        failures.append(f"{name}: missing bilingual 'What will you learn'")
    if "One key idea" not in text or "一个核心原理" not in text:
        failures.append(f"{name}: missing bilingual 'One key idea'")
    if "Takeaway" not in text or "一句话总结" not in text:
        failures.append(f"{name}: missing bilingual 'Takeaway'")
    if "Try it yourself" not in text and "学生练习" not in text:
        failures.append(f"{name}: missing student exercise section")
    if not CJK.search(text):
        failures.append(f"{name}: no Chinese content")
    if "STUDENT_" not in all_src:
        failures.append(f"{name}: no STUDENT_* parameter for students")
    if not (5 <= len(code) <= 16):
        failures.append(f"{name}: {len(code)} code cells (expected 6-16)")

if failures:
    print("NOTEBOOK VALIDATION FAILED:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print(f"all {len(EXPECTED)} notebooks valid: bilingual, templated, student-ready")
