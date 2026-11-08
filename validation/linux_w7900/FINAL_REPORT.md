# FINAL REPORT — YOLO26 Day 02 (v2.0.0)

Date: 2026-11-08 · Machine: 2× AMD EPYC 9334 (64C/128T, AuthenticAMD) ·
AMD Radeon PRO W7900 48 GB (gfx1100) · ROCm PyTorch 2.12.0+rocm7.14.1 ·
OpenVINO 2026.4.1 · NNCF 3.4.0 · Ultralytics 8.4.207

## A. 交付情况 / Deliverables

| Item | Status | Evidence |
|---|---|---|
| GitHub URL | PASS | https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs |
| HEAD SHA | PASS | `f904edb` (final HEAD; CI success on e7b684b and f904edb — report added in 933767a, doctor import-order fix in f904edb) |
| Release | PASS | v2.0.0 — https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs/releases/tag/v2.0.0 |
| CI (GitHub Actions) | PASS | `ci` workflow completed · success on HEAD |
| One-click install `setup.sh` | PASS | re-ran end-to-end; 10/10 steps; ALL HARD GATES PASS (AMD CPU + gfx1100 + ROCm torch + real compute + OpenVINO CPU) |
| Six notebooks | PASS | all 6 executed headlessly rc=0; bilingual; student-editable params; timings below |
| AMD CPU real verification | PASS | `AuthenticAMD`, 2× EPYC 9334; OpenVINO CPU plugin ran all CPU labs |
| W7900 real verification | PASS | gfx1100 (W7900D), `torch.version.hip=7.14.1`, GPU matmul + torchvision NMS smoke tests, real inference in Labs 00/A/D/E |
| Official model identity | PASS | yolo26l.pt, 53,211,173 B, sha256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`, 80-class COCO **detect** (verified at load, `weights/model_identity.json`) |
| Model / data never committed | PASS | only `weights/.gitkeep` + `data/manifest.json` in git |
| No training/fine-tuning anywhere | PASS | inference + export + PTQ only |

Notebook execution wall-times (quick defaults, cached exports; full log
`validation/linux_w7900/notebook_timings/timings.txt`):
00: 9 s · A: 14 s · B: 45 s · C: 31 s · D: 32 s · E: 12 s — comfortably inside
the class-time budget; discussion time is the bottleneck, not compute.

## B. 六组核心性能结论 / Six measured conclusions

Profile **full** (warmup 10 / 50 iterations; P50/P95 in ms; end-to-end per
image unless noted; mAP on the 120-image Educational COCO Subset):

| # | Experiment | Baseline | Optimization | Speed | Cost / Notes |
|---|---|---|---|---|---|
| 1 | CPU vs GPU (Lab A) | EPYC 9334 PyTorch 124.8 (P95 160.0) | W7900 ROCm 13.3 (P95 26.1) | **9.40×** | platform comparison (not pure runtime); GPU e2e incl. CPU pre/post |
| 2 | CPU runtime (Lab B) | PyTorch 154.6 | OpenVINO 50.1 | **3.09×** | mAP 0.6398 → 0.6349 (−0.005, parity); same CPU/weights/postproc; torch threads pinned to measured optimum (8) under 16-core cgroup quota; OV first-compile ≈ 1–2 s one-time |
| 3 | INT8 (Lab C) | OpenVINO FP32 49.5, 100 MB, mAP 0.6349 | NNCF MIXED INT8 28.2, 26 MB, mAP 0.5682 | **1.75×** speed, **3.8×** smaller | **ΔmAP −0.067** (PERFORMANCE preset worse: −0.106); quantize ≈ 5 min, 150 real calib images; honest finding: PTQ hurts YOLO26l accuracy here — production needs QAT/accuracy-aware flows; calibration verified bit-exact vs inference preprocessing |
| 4 | Resolution (Lab D-A) | imgsz 640: 13.2 ms, mAP 0.6398, VRAM 0.99 GB | imgsz 512: 12.5 ms, mAP 0.6159, VRAM 0.75 GB | −5% e2e latency | −0.024 mAP; forward-only scales more, but fixed pre/post dominate e2e at batch 1 — a measured nuance, not a failure |
| 5 | Batch (Lab D-B) | batch 1: 97.9 img/s, 10.2 ms/request | batch 8: 145.6 img/s, 54.9 ms/batch (6.9 ms/img equivalent) | **1.49×** throughput | VRAM peak 1.17 GB; batch time ÷ 8 explicitly labeled NOT request latency |
| 6 | Heterogeneous (Lab E) | serial 100 imgs: 72.3 img/s, 1.4 s | async CPU+GPU pipeline: 87.4 img/s, 1.1 s | **1.21×** | detections identical 100/100; per-image burst e2e p50 579 ms = queue-inclusive (serial has no queue) — throughput up, request wait under burst up; GPU already the bottleneck (infer ≈ 10 ms vs CPU prep ≈ 3 ms) so gain is honest but bounded |

Run-to-run variance: Lab A CPU P50 124.8 vs Lab B PyTorch 154.6 ms (~20%)
across separate processes — each lab therefore carries its own same-run
baseline; all raw samples retained (no outlier dropping).

## C. Notebook 学生体验 / Student experience

| NB | Understandable? | One-click? | Student-modifiable param | What students see | Theory vs result | No-gain explained? |
|---|---|---|---|---|---|---|
| 00 | yes, 8-step bilingual template | yes (9 s) | `STUDENT_IMAGE_INDEX` | real GPU detection w/ boxes; CPU/GPU/ROCm/OpenVINO versions; model sha256 | env verified before claims | n/a |
| 01 | yes | yes (14 s) | `STUDENT_DEVICE`, `STUDENT_ITERS` | CPU vs GPU P50/P95 table + bar chart | 9.4× matches parallel-math story | n/a (gain) |
| 02 | yes | yes (45 s) | `STUDENT_HINT` (LATENCY/THROUGHPUT, labeled extra) | 3.09× + mAP parity + first-compile cost | yes; AMD-validation caveat stated | n/a (gain) |
| 03 | yes | yes (31 s; first quantize ≈5 min pre-cached) | `STUDENT_CALIB_N`, prediction cells | size/latency/mAP three-panel chart; prediction vs measured | ΔmAP −0.067 **explained** (end-to-end head sensitivity, no AMX, PTQ limits) | partially-regrettable result turned into the lesson: INT8 is not free |
| 04 | yes | yes (32 s) | `STUDENT_IMGSZ_B`, batch cells | resolution trade-off + batch throughput/VRAM; latency semantics warning | −5% e2e explained via fixed overhead | yes — small gain explained |
| 05 | yes | yes (12 s) | `STUDENT_QUEUE_DEPTH`, `STUDENT_N_IMAGES` | serial vs pipeline wall-time, component breakdown, consistency check | 1.21× bounded gain explained via bottleneck (pre 3 ms < infer 10 ms); burst queue latency explained | yes |

## D. Slide 数据包 / Evidence pack

| Chart (assets/slides/, PNG+SVG) | Slides | Run ID / raw evidence |
|---|---|---|
| 00_environment.png | 28–32 | logs/env_report.json |
| 01_cpu_vs_gpu.png | 33 | labA_cpu_vs_gpu_{pytorch_cpu,pytorch_rocm_gpu}_20261108_*.csv (+_summary.json, sha256 embedded in chart) |
| 02_pytorch_vs_openvino_cpu.png | 34–38 | labB_runtime_cpu_{pytorch_cpu,openvino_cpu}_20261108_*.csv |
| 03_fp32_vs_int8.png | 39–42 | labC_int8_{openvino_fp32,openvino_int8}_20261108_*.csv |
| 04_resolution_vs_latency_accuracy.png | 43–46 | labD_resolution_imgsz{640,512}_20261108_*.csv |
| 05_batch_vs_throughput.png | 47–48 | labD_batch_batch{1,8}_forward_20261108_*.csv |
| 06_serial_vs_async_pipeline.png | 49–50 | results/latest.json → labE.per_image |

Mapping doc: `docs/SLIDE_28_51_EVIDENCE_MAP.md` (also lists the six headline
conclusions for Slide 51). Charts regenerate from CSV/JSON via
`course/charts.py` / `scripts/run_experiments.py`.

## E. 剩余问题 / Remaining items

| Item | State | Detail |
|---|---|---|
| setup / doctor / run_all quick+full | PASS | logs in `logs/` (not committed) |
| pytest | PASS | 12/12 (6 unit incl. CI-safe skips + 6 hardware integration) |
| Notebooks executed | PASS | 6/6 rc=0 |
| Charts from real data | PASS | 7 PNG+SVG with run-id + CSV sha256 footnotes |
| GitHub CI | PASS | success on HEAD e7b684b |
| Release | PASS | v2.0.0 (full acceptance) |
| OpenVINO-on-AMD official certification | DISCLOSED | OpenVINO's official validation on AMD CPUs is limited; all claims here are machine-measured — stated in README, Lab B, protocol |
| INT8 ΔmAP −0.067 | HONEST RESULT | best of {PERFORMANCE −0.106, MIXED −0.067, head-ignored −0.101, SmoothQuant −0.067}; documented as the accuracy price of PTQ here |
| Full-profile multi-repetition study | NOT_RUN | one full pass per lab (50 iters each) + repeated quick passes; multi-run statistical study left as future work |
| Lab A CPU-vs-Lab B PyTorch 20% cross-run variance | NOTED | separate processes; per-lab baselines used everywhere |

**Bottom line**: one official YOLO26l, two AMD devices, six bilingual
notebooks, six real experiments, one reproducible open-source course —
delivered, measured, published.
