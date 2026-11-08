# Notebook Execution Matrix (v2.0.1)

Executed: 2026-11-08 11:29:32 on this repo's
kernel via nbclient (real execution, not structural validation).
Machine: 2x AMD EPYC 9334 · Radeon PRO W7900D (gfx1100) · ROCm torch.

**6/6 notebooks PASS · 45/45 code cells executed**

| Notebook | Status | Code cells | Wall (s) | Kernel | Expected markers |
|---|---|---|---|---|---|
| 00_AMD_Environment_and_YOLO26l.ipynb | PASS | 7/7 | 7.0 | yolo26-day2 | sha256:OK, AuthenticAMD:OK |
| 01_Lab_A_CPU_vs_GPU_Baseline.ipynb | PASS | 9/9 | 12.9 | yolo26-day2 | CPU P50:OK, GPU P50:OK |
| 02_Lab_B_OpenVINO_AMD_CPU.ipynb | PASS | 7/7 | 39.3 | yolo26-day2 | PyTorch CPU P50:OK, OpenVINO CPU P50:OK |
| 03_Lab_C_OpenVINO_INT8.ipynb | PASS | 9/9 | 29.6 | yolo26-day2 | FP32 P50:OK, INT8 P50:OK, mAP:OK |
| 04_Lab_D_ROCm_GPU_Optimization.ipynb | PASS | 6/6 | 25.0 | yolo26-day2 | imgsz:OK, batch=1:OK, batch=8:OK |
| 05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb | PASS | 7/7 | 10.4 | yolo26-day2 | serial:OK, pipeline:OK |

Executed notebook copies (with outputs) are saved next to this file under `executed/`.

Model rule gate: every notebook statically checked for the official yolo26l reference and absence of smaller/fake models (yolo26n/s/m/x, best.pt, random tensors).
