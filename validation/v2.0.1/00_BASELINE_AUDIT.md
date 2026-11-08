# 00 — Baseline Audit (v2.0.1 validation campaign start)

Date: 2026-11-08 · Auditor: independent release-audit pass (R1–R4 mission)
Branch created from `origin/main`: `release/v2.0.1-r1-r3-validated`

## 1. Repository state at freeze

| Item | Value |
|---|---|
| origin/main HEAD at fetch | `bcf4f75b16291bc17141cf11dfda1bda1d0ca4a4` |
| Working tree | clean (0 modified files) |
| v2.0.0 tag SHA | `e7b684bef6923a9eb0ebf5861b30a469fb5c3cc6` (= last full-review commit) |
| Commits after v2.0.0 tag | `933767a` (final report + slide map), `f904edb` (doctor import-order fix), `bcf4f75` (record final HEAD) |
| Open PRs | 0 |
| GitHub Actions (`ci`) | success on `e7b684b`, `933767a`, `f904edb`, `bcf4f75` |
| v2.0.0 Release | published 2026-11-08T10:13:24Z, target main, non-draft; body references `validation/linux_w7900/FINAL_REPORT.md` (path EXISTS in repo — verified) |

## 2. The six notebooks (v2.0.0 state)

| Notebook | md cells | code cells | student knobs |
|---|---|---|---|
| 00_AMD_Environment_and_YOLO26l.ipynb | 6 | 7 | STUDENT_IMAGE_INDEX … |
| 01_Lab_A_CPU_vs_GPU_Baseline.ipynb | 8 | 7 | STUDENT_DEVICE, STUDENT_ITERS … |
| 02_Lab_B_OpenVINO_AMD_CPU.ipynb | 8 | 7 | STUDENT_HINT … |
| 03_Lab_C_OpenVINO_INT8.ipynb | 8 | 8 | STUDENT_CALIB_N … |
| 04_Lab_D_ROCm_GPU_Optimization.ipynb | 6 | 6 | STUDENT_IMGSZ_B, batch … |
| 05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb | 8 | 6 | STUDENT_QUEUE_DEPTH, STUDENT_N_IMAGES … |

Structural validation (`tests/validate_notebooks.py`) passes in CI, but CI never
EXECUTES the notebooks — that is R1.4 scope.

## 3. v2.0.0 benchmark numbers (HISTORICAL — must be re-measured this round)

| Experiment | v2.0.0 claim | Evidence at the time |
|---|---|---|
| Lab A CPU vs GPU | 124.8 → 13.3 ms P50, **9.40×** | results/raw/labA_cpu_vs_gpu/*.csv |
| Lab B PyTorch vs OpenVINO CPU | 154.6 → 50.1 ms P50, **3.09×**, ΔmAP −0.005 | results/raw/labB_runtime_cpu/*.csv |
| Lab C FP32 vs INT8 | 49.5 → 28.2 ms P50, **1.75×**, size 3.8×, ΔmAP **−0.0667** | results/raw/labC_int8/*.csv |
| Lab D1 resolution 640→512 | 13.2 → 12.5 ms, ΔmAP −0.024 | results/raw/labD_resolution/*.csv |
| Lab D2 batch 1→8 | 97.9 → 145.6 img/s, **1.49×** | results/raw/labD_batch/*.csv |
| Lab E serial vs async | 72.3 → 87.4 img/s, **1.21×** | results/latest.json labE |

None of these numbers may be reused for v2.0.1 public claims without a fresh
run on this round's final code (Mission §0.5).

## 4. Defects found in v2.0.0 code (audit reproduction)

| ID | Severity | Location | Defect |
|---|---|---|---|
| B-01 | P0 | scripts/setup.sh:144-149 | `data/manifest.json` IS committed; images are NOT. Fresh clone has manifest-only → step 7 skips `prepare_data.py` → no images → false PASS of "data ready". |
| B-02 | P0 | course/benchmark.py:195-198 | `save_run()` writes summary.json, hashes it, then REWRITES the file with the hash inside → recorded `json_sha256` never matches file bytes. |
| B-03 | P0 | course/pipeline.py:209-224 | `compare_results()` compares only `len(boxes)` per image; no class/box/conf/IoU parity, cannot detect swapped or shifted detections. |
| B-04 | P0 | (missing) | No notebook EXECUTION gate: `run_all.sh` runs the experiment script; `tests/validate_notebooks.py` checks JSON structure only. |
| B-05 | P0 | course/quantization.py:64-65 | INT8 cache hit is "xml exists" only — a 30-image calibration request silently reuses the 150-image model. No cache identity. |
| B-06 | P1 | course/setup_checks.py:36-37 | `threads` computed as threads_per_core × sockets (=4) instead of total logical CPUs (=128); no cgroup quota (16 cores), affinity, NUMA topology recorded. |
| B-07 | P1 | results/latest.json | `run_experiments.main()` carries previous-run dicts forward on partial reruns → cross-run contamination; labC `quantization_time_s=0.2` in latest.json proves a cache hit was recorded as "quantization". |
| B-08 | P1 | history | OpenVINO CPU latency bimodality (~30/70 ms alternating samples in labB/labC CSVs) undiagnosed; run-to-run CPU variance ~20% (Lab A 124.8 vs Lab B 154.6 ms) not investigated. |
| B-09 | P2 | (missing) | No independent audit script that recomputes P50/P95/speedups from raw CSVs. |
| B-10 | P2 | (missing) | No SHA256SUMS / no tamper-evident campaign verification command. |

## 5. This round's REAL hardware (re-confirmed, not copied from history)

```
CPU vendor        : AuthenticAMD  (grep vendor_id /proc/cpuinfo)
CPU model         : AMD EPYC 9334 32-Core Processor
Sockets / Cores   : 2 sockets × 32 cores = 64 physical cores, 2 threads/core = 128 logical
NUMA              : 2 nodes (node0: 0-31,64-95 · node1: 32-63,96-127)
Cgroup CPU quota  : 1600000/100000 = 16 cores (container quota ≠ physical topology)
Affinity          : 0-127 visible to process
Instruction sets  : avx2, avx512f, avx512_bf16, … (NO Intel AMX — AMD Zen4)
GPU               : AMD Radeon Pro W7900D (torch device name), gfx1100 (rocminfo),
                    PCI device 0x744b, 1 GPU visible to ROCm runtime (8 on host bus)
ROCm kernel       : ROCk 6.16.13 loaded; rocm-smi available
PyTorch           : 2.12.0+rocm7.14.1 (hip 7.14.60850) — matches user pin
torchvision       : 0.27.0+rocm7.14.1 · torchaudio 2.11.0+rocm7.14.1
OpenVINO          : 2026.4.1 · NNCF 3.4.0 · Ultralytics 8.4.207
RAM / disk        : 503 GB RAM · /workspace 85 GB free
yolo26l.pt        : 53,211,173 B · sha256 9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26
                    80-class COCO detect (verified via ultralytics load)
```

## 6. Validation plan for this round

1. **R1** — fix B-01..B-04 (+10): verified data integrity & incremental repair;
   tamper-evident evidence chain (CSV → summary → SHA256SUMS → `verify_evidence.py`);
   per-detection Lab E parity (class/xyxy/conf, one-to-one IoU matching);
   real notebook execution gate (`scripts/validate_notebooks.sh --execute`, nbclient);
   cold-start install in a fresh clone + fresh venv (R1.1 acceptance).
2. **R2** — fix B-05..B-07; diagnose B-08 (interleaved A/B runs, thread/stream
   control, OpenVINO config dump); re-run ALL six experiments on this machine into
   `validation/v2.0.1/campaigns/<run_id>/` (fresh data only); independent audit
   script recomputing every public number from CSVs; regenerate 6 charts + SVG.
3. **R3** — notebook teaching upgrades (bilingual 10-step template, Amdahl
   interactive cell, real OpenVINO runtime demo, INT8 numeric example, batch /
   resolution timeline semantics, pipeline real timeline from stage records);
   instructor guide covering slides 28–51; student-experience review with scores.
4. **R4** — freeze candidate SHA; independent fresh-clone audit with fault
   injection (tamper CSV / delete image / alter weight SHA / swap detection /
   change INT8 calibration / kernel mismatch / stale run-id mixing); four gates;
   PR → CI → merge → tag `v2.0.1` → GitHub Release.

Historical v2.0.0 artifacts are PRESERVED (nothing deleted or overwritten);
new evidence lives under `validation/v2.0.1/`.
