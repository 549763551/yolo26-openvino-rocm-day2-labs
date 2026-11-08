# FINAL REPORT — YOLO26 Day 02 v2.0.1 (Independent Audit & Release Validation)

Date: 2026-11-08 · Campaign: `validation/v2.0.1/campaigns/c20261108_1108`
Machine: 2× AMD EPYC 9334 (64C/128T, AuthenticAMD; container cgroup quota 16
cores) · AMD Radeon PRO W7900D 48 GB (gfx1100) · ROCm PyTorch
2.12.0+rocm7.14.1 (HIP 7.14.60850) · OpenVINO 2026.4.1 · NNCF 3.4.0 ·
Ultralytics 8.4.207 · official yolo26l.pt
(53,211,173 B, sha256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`,
80-class COCO detect).

Shared-host disclosure: benchmarks ran on a shared machine (load average
28–42 recorded per round in `CAMPAIGN.json → round_env`). No exclusivity is
claimed; medians of 3 rounds are reported with per-round spread.

---

## A. Release status

| Item | Value |
|---|---|
| Repo | https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs |
| Release branch | `release/v2.0.1-r1-r3-validated` (head `e5af233`) |
| PR | https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs/pull/1 |
| CI | GitHub Actions `ci` — success on the release branch (unit tests, notebook structure gate, protocol files, offline chart re-render job) |
| Tag / Release | `v2.0.1` — published after merge (see GitHub Releases) |
| v2.0.0 tag (historical) | `e7b684bef6923a9eb0ebf5861b30a469fb5c3cc6` |
| Gates | Engineering **GO** · Benchmark **GO** · Teaching **GO** · Open Source **GO** (see INDEPENDENT_AUDIT_LOG.md) |

## B. R1 — Release integrity & reprocibility

| Item | Result | Evidence |
|---|---|---|
| B-01 data-skip defect | FIXED — setup verifies real files (counts/labels/yaml/calib-list/SHA256/isolation) and repairs incrementally; negative tests (manifest-only clone must FAIL) in `tests/test_v201_fixes.py` | `scripts/verify_data.py`, `scripts/setup.sh` step 7 |
| Cold-start install (fresh clone + fresh venv) | PASS — 10/10 steps; torch pins verified `2.12.0+rocm7.14.1`/`0.27.0`/`2.11.0`; **disclosure**: ROCm wheels served from local pip cache (transport cache; index contacted for resolution), yolo26l.pt + COCO subsets real network downloads; data step detected manifest-only state and repaired via prepare_data → verify PASS | `validation/v2.0.1/INDEPENDENT_AUDIT_LOG.md` (cold-start summary; raw installer transcript was **not committed**) |
| B-02 self-hash defect | FIXED — evidence chain CSV→summary→SHA256SUMS; `verify_evidence.py` recomputes from bytes; tamper tests (CSV byte change → FAIL; summary stats change → FAIL) pass | `course/benchmark.py`, `tests/test_v201_fixes.py` |
| B-03 count-only Lab E | FIXED — per-detection parity (class+IoU≥0.95+conf±0.02); this campaign measured **100/100 images bit-identical (min IoU 1.0, max Δconf 0.0)** | `course/pipeline.py`, campaign `labE_pipeline_compare_*.json` |
| B-04 notebook execution gate | **6/6 notebooks PASS · 45/45 code cells** (nbclient real execution, expected-output markers, model-rule static gate) | `validation/v2.0.1/notebooks/EXECUTION_MATRIX.md` + `executed/*.ipynb` |
| Release path references | all referenced release paths exist, except the former uncommitted cold-start raw transcript (now explicitly disclosed) (v2.0.0 release body path `validation/linux_w7900/FINAL_REPORT.md` verified present; v2.0.1 release references `validation/v2.0.1/FINAL_REPORT.md` — this file) | checked in R4 |

## C. R2 — Benchmark scientific revalidation (campaign c20261108_1108)

All six experiments re-executed on real hardware, 3 rounds × 50 samples per
latency variant (warm-up 10 excluded), median-of-rounds published. mAP
evaluations once per model configuration (deterministic). INT8 REBUILT fresh
in-campaign (NNCF MIXED, 150 real train2017 images, 44 s, identity key in
`CACHE_KEY.json`).

| 实验 | 原始方案 | 优化方案 | 新实测收益 | 精度/资源代价 |
|---|---|---|---|---|
| CPU vs GPU (Lab A, e2e) | EPYC 9334 PyTorch P50 191.3 ms | W7900D ROCm P50 13.5 ms | **14.14×**（P95 **470.6→15.7**，由逐轮原始 CSV 重算更正） | 平台对比（非软件优化）；轮间波动 7.5%/5.8% |
| CPU Runtime (Lab B) | PyTorch 198.5 ms | OpenVINO 87.2 ms | **2.28×** | mAP 0.6398→0.6349（Δ−0.0049，一致）；首次编译 1–2 s |
| CPU INT8 (Lab C) | FP32 98.3 ms | INT8 78.1 ms | **1.26×**（本轮 P50）；文件大小历史约 3.8×，**非本轮复测** | **ΔmAP −0.0667（0.6349→0.5682，明显代价）** |
| GPU Resolution (Lab D-A) | imgsz 640: 13.68 ms · mAP .6398 | imgsz 512: 13.50 ms · mAP .6159 | **−1.3% e2e**（batch 1 固定开销主导） | ΔmAP −0.024；**本轮未保存 GPU allocator 显存数据，不可引用旧显存数字作新实测** |
| GPU Batch (Lab D-B, forward) | batch 1: 95.2 img/s | batch 8: 144.9 img/s | **1.52×** | 55.2 ms/批÷8=6.9 ms 仅等效成本；e2e 13.6 ms/req → 80.0 ms/8 张请求 |
| CPU/GPU Pipeline (Lab E) | serial 58.6 img/s | async 83.1 img/s | **1.42×**（100 张突发） | 逐目标一致 100/100（IoU 1.0，Δconf 0.0）；流水线突发 e2e P50 603.5 ms（三轮 P50 中位数） 含排队（与串行 14.0 ms 口径不同，分开报告） |

Per-round raw data, spread and scope notes: `CAMPAIGN.json` (aggregate +
round_env). Independent audit: `AUDIT_REPORT.md` — **53 runs recomputed from
CSV bytes, 96 files hash-verified, all headline ratios/mAP/parity checks
PASS** (`scripts/audit_benchmarks.py`).

**Why numbers differ from v2.0.0** (disclosed, not hidden): v2.0.1 pinned CPU
variants to 16 CPUs matching the cgroup quota, eliminating throttling (the
v2.0.0 bimodality — root cause proven in
`validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.{md,json}`: OV's default 64 threads
vs 16-core quota → cgroup throttle counters match the slow samples exactly);
shared-host load is higher this round and is now recorded; multi-round
medians replace single-pass numbers. INT8 ΔmAP (−0.0667) reproduced exactly
(deterministic PTQ).

Unexplained residuals: labB_ov first recorded round P50 99.4 ms vs next two ≈ 87 ms
(neighbor drift on shared cores; recorded); Lab D-A e2e difference 640 vs 512
is within measurement noise at batch 1 (forward-only compute does scale,
documented).

## D. R3 — Teaching improvements

All six notebooks upgraded (bilingual 10-section template: objective → key
idea → **predict** → run → change one thing → compare → why → try it →
**troubleshooting** → takeaway) and RE-EXECUTED 6/6 · 45/45 cells.

| NB | What changed (v2.0.1) | Student learns | Interactive | Slide evidence | Exec time |
|---|---|---|---|---|---|
| 00 | dual-path architecture diagram (OpenVINO-CPU vs ROCm-GPU); troubleshooting | which runtime owns which device; identity-first experiments | STUDENT_IMAGE_INDEX re-detects | 28–32 | 7 s |
| 01 | Amdahl mini-lab (20+10→25 ms, hand-computed) + measured model-only vs e2e gap | fixed overhead caps speedup | guess variable + STUDENT_DEVICE/ITERS | 30, 33 | 13 s |
| 02 | real 5-step `ov.Core()` demo; fixed no-op STUDENT_HINT knob; AMX claim removed | runtime = different execution engine, not magic | STUDENT_HINT compiles differently (verified) | 34–38 | 39–56 s |
| 03 | five-number INT8 example (scale/zero-point/error); cache identity; deployment question | quantization error is concrete; ΔmAP has business meaning | STUDENT_CALIB_N 30↔150 separate caches (verified) | 39–42 | 30 s |
| 04 | resolution causal chain + batch timeline ASCII; equivalent-cost vs request-latency labels | compute ∝ imgsz²; batch semantics | STUDENT_IMGSZ_B re-measures (verified) | 43–48 | 25 s |
| 05 | measured stage-timeline chart; two-scenario latency framing; 4 discussion questions | overlap must be proven; scopes differ | STUDENT_QUEUE_DEPTH re-runs (verified) | 49–50 | 10 s |

Instructor guide: `docs/INSTRUCTOR_GUIDE.md` (slides 28–51, per-slide
knowledge, interactions, answers, misconceptions, timings, forbidden claims).
Student review: `validation/v2.0.1/STUDENT_EXPERIENCE_REVIEW.md` (engineer-
simulated; all dimensions 4.6–5.0/5; explicitly NOT a real user study).
Knob verification: `validation/v2.0.1/STUDENT_KNOB_VERIFICATION.md` — 5/5
student parameters verifiably change measured outputs.

## E. R4 — Independent release audit

Performed in a **fresh clone** of the frozen candidate lineage (`ca344b3` +
audit-tooling fix `e5af233`) at `/workspace/audit_v201` — full transcript and
gate verdicts in `validation/v2.0.1/INDEPENDENT_AUDIT_LOG.md`.

Battery: verify_data PASS · doctor (sha-from-bytes) ALL GATES PASS ·
verify_evidence PASS · independent benchmark audit PASS (53 runs / 96 files)
· 19/19 unit tests · model identity MATCH · README-vs-campaign numbers MATCH
· **notebook execution 6/6 · 45/45 cells** (with fresh export + fresh INT8
quantization in the audit tree) · open-source hygiene (0 secrets in history,
no artifacts committed).

Fault injections — **all 9 detected**:

| Fault injected | Expected | Observed |
|---|---|---|
| modify one CSV byte | FAIL | DETECTED |
| modify summary JSON stats | FAIL | DETECTED |
| delete one val image | FAIL | DETECTED |
| alter recorded weight sha | FAIL | DETECTED |
| swap one detection box/class | FAIL | DETECTED |
| INT8 calibration 150→30 | separate cache | VERIFIED (identity-keyed) |
| stale CACHE_KEY | rebuild, no stale reuse | VERIFIED |
| mix stale run id into campaign | FAIL | DETECTED |
| unknown notebook kernel | FAIL | DETECTED |

Independence disclosure: the audit used a fresh clone and its own
kernel/venv; weights/data reached it via verified transport cache (sha256
re-verified from bytes); OpenVINO export and INT8 quantization were rebuilt
inside the audit tree rather than copied.

## F. Final teaching outcomes

- **Students can run it**: one-click setup (verified in a fresh clone + fresh
  venv, data step auto-repairs), six notebooks executed 45/45 cells, five
  student knobs verifiably change real outputs.
- **Teachers can explain it**: instructor guide covers slides 28–51 with the
  measured numbers, the why, the costs, the misconceptions and the claims
  that must not be made; every chart carries campaign/run ids and evidence
  paths.
- **Community can verify it**: git SHA → model/data SHA256 → immutable CSVs →
  SHA256SUMS → `verify_evidence.py` / `audit_benchmarks.py` (independent
  recomputation) → charts → this report. No step trusts recorded values
  without recomputation.

Simple for Students. Rigorous for Engineers. Reproducible for Everyone.
