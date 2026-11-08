#!/usr/bin/env python3
"""Execute the six notebooks for real (nbclient) and verify their outputs.

v2.0.1 (fixes defect B-04): v2.0.0 validated notebook JSON structure only.
This executor:
  * opens every notebook with nbformat;
  * statically gates the MODEL RULE (official yolo26l only — see
    FORBIDDEN_PATTERNS; anything smaller/fake fails the gate);
  * runs EVERY code cell in a fresh kernel via nbclient (same interpreter);
  * fails on the first raising cell (allow_errors=False);
  * checks expected per-notebook output markers actually appeared;
  * saves the executed notebook + a per-notebook log under
    validation/<version>/notebooks/executed/;
  * writes EXECUTION_MATRIX.md (N/N notebooks, N/N code cells).

Exit code 0 only when all notebooks pass with all expected markers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NOTEBOOKS = [
    "00_AMD_Environment_and_YOLO26l.ipynb",
    "01_Lab_A_CPU_vs_GPU_Baseline.ipynb",
    "02_Lab_B_OpenVINO_AMD_CPU.ipynb",
    "03_Lab_C_OpenVINO_INT8.ipynb",
    "04_Lab_D_ROCm_GPU_Optimization.ipynb",
    "05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb",
]

# Model rule (mission §0.1): official yolo26l only. These static patterns
# catch a swapped/smaller/fake model before wasting an execution.
FORBIDDEN_PATTERNS = [
    r"yolo(26|11|8)?[nsmx]\.pt",      # nano/small/medium/x-large variants
    r"\bbest\.pt\b",                   # Day-01 fine-tuned weights
    r"\blast\.pt\b",
    r"torch\.randn.*boxes",            # fake detections
    r"np\.random.*boxes",
]
REQUIRED_PATTERN = re.compile(
    r"yolo26l|load_official_yolo26l|openvino_fp32_dir|int8_cache_dir|"
    r"quantize_int8|Yolo26lStages", re.I)

# Output markers proving the notebook produced its key REAL results.
EXPECTED_MARKERS = {
    "00_AMD_Environment_and_YOLO26l.ipynb": ["sha256", "AuthenticAMD"],
    "01_Lab_A_CPU_vs_GPU_Baseline.ipynb": ["CPU P50", "GPU P50"],
    "02_Lab_B_OpenVINO_AMD_CPU.ipynb": ["PyTorch CPU P50", "OpenVINO CPU P50"],
    "03_Lab_C_OpenVINO_INT8.ipynb": ["FP32 P50", "INT8 P50", "mAP"],
    "04_Lab_D_ROCm_GPU_Optimization.ipynb": ["imgsz", "batch=1", "batch=8"],
    "05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb": ["serial", "pipeline"],
}


def outputs_text(nb) -> str:
    out = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o.get("text", [])))
            elif "data" in o:
                d = o["data"]
                out.append("".join(d.get("text/plain", [])))
    return "\n".join(out)


def static_model_gate(name: str, src: str) -> list[str]:
    errs = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, src):
            errs.append(f"{name}: forbidden pattern {pat!r} (model rule)")
    if not REQUIRED_PATTERN.search(src):
        errs.append(f"{name}: no reference to the official yolo26l path")
    return errs


def pick_kernel(nb, available: set[str]) -> str:
    md = nb.metadata.get("kernelspec", {}).get("name", "")
    if md in available:
        return md
    for cand in ("yolo26-day2", "python3"):
        if cand in available:
            return cand
    raise SystemExit(f"no usable kernel (have {sorted(available)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "validation" / "v2.0.1"
                                             / "notebooks"))
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-cell timeout in seconds")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    exec_dir = out_dir / "executed"
    exec_dir.mkdir(parents=True, exist_ok=True)

    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    available = set(KernelSpecManager().find_kernel_specs().keys())

    results = []
    all_errors: list[str] = []
    for name in NOTEBOOKS:
        path = REPO / "notebooks" / name
        rec = {"notebook": name, "status": "PASS", "code_cells": 0,
               "cells_ok": 0, "wall_s": 0.0, "markers": {}, "error": None}
        t0 = time.time()
        try:
            nb = nbformat.read(str(path), as_version=4)
            src_all = "\n".join("".join(c.source) for c in nb.cells
                                if c.cell_type == "code")
            gate = static_model_gate(name, src_all)
            if gate:
                raise RuntimeError("; ".join(gate))
            kernel = pick_kernel(nb, available)
            rec["kernel"] = kernel
            client = NotebookClient(
                nb, timeout=args.timeout, kernel_name=kernel,
                resources={"metadata": {"path": str(REPO)}},
                allow_errors=False)
            client.execute()
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            rec["code_cells"] = len(code_cells)
            executed = sum(1 for c in code_cells
                           if c.get("execution_count") is not None)
            rec["cells_ok"] = executed
            text = outputs_text(nb)
            missing = [m for m in EXPECTED_MARKERS[name] if m not in text]
            rec["markers"] = {m: (m in text) for m in EXPECTED_MARKERS[name]}
            if executed != len(code_cells):
                raise RuntimeError(f"only {executed}/{len(code_cells)} cells "
                                   "have execution counts")
            if missing:
                raise RuntimeError(f"expected output markers missing: {missing}")
            nbformat.write(nb, str(exec_dir / name))
        except Exception as e:
            rec["status"] = "FAIL"
            rec["error"] = str(e)[:500]
            all_errors.append(f"{name}: {e}")
        rec["wall_s"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"[{'PASS' if rec['status']=='PASS' else 'FAIL'}] {name} "
              f"({rec['wall_s']}s, {rec['cells_ok']}/{rec['code_cells']} cells)"
              + (f" — {rec['error']}" if rec['error'] else ""), flush=True)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    cells_total = sum(r["code_cells"] for r in results)
    cells_ok = sum(r["cells_ok"] for r in results if r["status"] == "PASS")

    lines = [
        "# Notebook Execution Matrix (v2.0.1)",
        "",
        f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S')} on this repo's",
        "kernel via nbclient (real execution, not structural validation).",
        "Machine: 2x AMD EPYC 9334 · Radeon PRO W7900D (gfx1100) · ROCm torch.",
        "",
        f"**{n_pass}/{len(results)} notebooks PASS · {cells_ok}/{cells_total} "
        "code cells executed**",
        "",
        "| Notebook | Status | Code cells | Wall (s) | Kernel | Expected markers |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        marks = ", ".join(f"{k}:{'OK' if v else 'MISSING'}"
                          for k, v in r.get("markers", {}).items()) or "-"
        lines.append(f"| {r['notebook']} | {r['status']} | "
                     f"{r['cells_ok']}/{r['code_cells']} | {r['wall_s']} | "
                     f"{r.get('kernel','-')} | {marks} |")
    if any(r["status"] == "FAIL" for r in results):
        lines += ["", "## Failures", ""]
        lines += [f"- {r['notebook']}: {r['error']}"
                  for r in results if r["status"] == "FAIL"]
    lines += ["", "Executed notebook copies (with outputs) are saved next to "
              "this file under `executed/`.", "",
              "Model rule gate: every notebook statically checked for the "
              "official yolo26l reference and absence of smaller/fake models "
              "(yolo26n/s/m/x, best.pt, random tensors)."]
    (out_dir / "EXECUTION_MATRIX.md").write_text("\n".join(lines) + "\n")
    (out_dir / "execution_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{n_pass}/{len(results)} notebooks PASS · {cells_ok}/{cells_total} "
          f"code cells -> {out_dir / 'EXECUTION_MATRIX.md'}")
    return 0 if n_pass == len(results) and all_errors == [] else 1


if __name__ == "__main__":
    sys.exit(main())
