<<<<<<< HEAD
# Slides 28–51 — v2.0.1 verified evidence map

**Authoritative source:** `validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json`, audited in `AUDIT_REPORT.md`. Branch `release/v2.0.1-r1-r3-validated`. The previous `results/latest.json` and `results/RESULTS_TABLE.md` were v2.0.0 historical records; this file and updated RESULTS_TABLE take precedence for v2.0.1 slides.

**Measurement contract:** 3 independent rounds × 50 timed iterations per latency variant (10 warmups excluded); report the **median of round-level P50s**, not a single-run maximum. Shared Linux machine: 2 × AMD EPYC 9334 (64 physical cores, **16-core cgroup CPU quota**), Radeon PRO W7900D (gfx1100), ROCm PyTorch 2.12.0+rocm7.14.1, OpenVINO 2026.4.1, Ultralytics 8.4.207. Official YOLO26l model SHA256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`.
=======
# Slide 28–51 Evidence Map / 幻灯片证据映射 (v2.0.1)

Every chart below was generated from **real measured data on the lab machine**
(2× AMD EPYC 9334 · Radeon PRO W7900D (gfx1100) · torch 2.12.0+rocm7.14.1 ·
OpenVINO 2026.4.1 · NNCF 3.4.0 · Ultralytics 8.4.207 · official yolo26l.pt,
sha256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`).

**v2.0.1 evidence source**: formal campaign `validation/v2.0.1/campaigns/c20261108_1108/`
— 3 independent rounds × 50 timed samples per latency variant (warm-up 10,
excluded), median-of-rounds P50/P95, every sample kept, every file covered by
SHA256SUMS and independently re-audited (`scripts/audit_benchmarks.py` →
`AUDIT_REPORT.md`: 53 runs recomputed, 96 files hashed, PASS). Charts embed
the campaign id, run ids and evidence paths in their footnotes.

Historical v2.0.0 numbers remain archived under `results/raw/` — they are NOT
reused for v2.0.1 claims (mission §0.5).
>>>>>>> b662414 (R2.7 evidence map + R3.8 instructor guide (slides 28-51); README/README.en/results table with campaign c20261108_1108 numbers; 6/6 notebooks executed 45/45 cells)

Every slide should carry the scope (P50 end-to-end vs forward-only vs burst-throughput), source campaign ID, and a brief caveat. **Do not use v2.0.0 numbers as current measurements.**

<<<<<<< HEAD
| Slide | Suggested learning goal | Verified v2.0.1 evidence | Asset |
|---|---|---|---|
| 28 | From Day1 training to Day2 deployment | Same YOLO26l, different execution engines | `00_environment.png` |
| 29 | Three levers: less/faster/parallel compute | Amdahl + six labs | conceptual diagram |
| 30 | Where latency lives | preprocess → forward → postprocess → queue | measured component scopes |
| 31 | Measurement is an experiment | warmup 10, 50 samples × 3 rounds | `CAMPAIGN.json` |
| 32 | Hardware is not the CPU budget | 2×EPYC but 16-core cgroup; W7900D gfx1100 | `ENV.json` + `00_BASELINE_AUDIT.md` |
| 33 | Lab A: CPU vs GPU | CPU **191.28ms** → W7900 **13.53ms** P50 (**14.14×**) | `01_cpu_vs_gpu.png` |
| 34 | Why GPU parallelism helps | parallel convolutions + CPU pre/post | explanatory diagram |
| 35 | Lab B: same CPU, different runtime | PyTorch **198.54ms** → OpenVINO **87.22ms** (**2.28×**) | `02_pytorch_vs_openvino_cpu.png` |
| 36 | Runtime transformation vs hardware change | Same CPU, same model family; backend execution change | pipeline diagram |
| 37 | Preserve evaluation accuracy | mAP50–95 **0.6398 → 0.6349** (Δ **−0.0049**) | campaign map JSON |
| 38 | CPU load, cache, NUMA & quota | OpenVINO round P50 86.79–99.40ms; 16-core quota | campaign rounds and diagnosis |
| 39 | Lab C: precision/quantization | FP32 **98.30ms** → INT8 **78.14ms** (**1.26×**) | `03_fp32_vs_int8.png` |
| 40 | Quantization math | FP32 vs INT8 value ranges and scale/zero-point; NNCF MIXED | educational diagram |
| 41 | Cost of INT8 | mAP **0.6349 → 0.5682**; signed Δ **−0.0667** | corrected INT8 chart |
| 42 | When *not* to deploy INT8 | Accuracy constraints, PTQ vs QAT, calibration | decision matrix |
| 43 | Lab D: reduce input size | 640²→512² input elements −36%; **not** E2E speedup | geometric illustration |
| 44 | Real resolution result | P50 **13.68ms → 13.50ms**, about **1.3% faster** | `04_resolution_vs_latency_accuracy.png` |
| 45 | Amdahl: bottleneck migration | 512 P95≈20.04ms vs 640 P95≈14.96ms; run variation | bottleneck bars |
| 46 | Small-object cost | mAP **0.6398 → 0.6159** (Δ −0.0239) | measured map and example |
| 47 | Batch is about throughput | Forward-only **95.2→144.9 img/s**, **1.52×** | `05_batch_vs_throughput.png` |
| 48 | Throughput ≠ latency | b1 **10.51ms/forward**; b8 **55.22ms/batch**; E2E **13.58ms vs 80.04ms** | timeline |
| 49 | Lab E: asynchronous pipeline | Serial **58.62→83.13 img/s** (**1.42×**) | `06_serial_vs_async_pipeline.png` |
| 50 | Queueing can dominate wait | Serial P50≈13.96ms (no queue) vs burst pipeline≈610.81ms (queue-inclusive); non-equivalent arrival scenarios | queue timeline |
| 51 | Engineering summary & interactive exam | Six measured outcomes, accuracy & latency tradeoffs, methodology | summary grid |

## Guardrails and corrections

1. **INT8 ΔmAP is negative**: `0.5681614 - 0.6349081 = −0.0667467`. Corrected generator must display `−0.0667`, never `+0.0667`.
2. **No fabricated VRAM**: The v2.0.1 resolution and batch summary variants do not provide allocator memory readings. The figures must not display `0.00 GB`. Use E2E batch duration as the paired batch chart instead.
3. **Model sizes**: FP32 ≈100MB / INT8 ≈26MB were recorded in v2.0.0. New slide speed and accuracy numbers come from v2.0.1; do not label model file sizes as remeasured in the new campaign.
4. **Lab E**: The full-burst pipeline latency P50 (610.81ms) includes arrival queue time; serial P50 (13.96ms) has no submission queue. It is **not a like-for-like per-request latency comparison**; headline is 1.42× throughput.
5. **External validity**: CPU timing is influenced by 16-core cgroup quota, noisy shared-host load and throttling. Ratios describe this exact environment. mAP is on a 120-image educational COCO subset, **not** the official COCO benchmark.
6. **Environment legacy artifact**: Campaign `ENV.json` has topology zeros due to the previously faulty `Socket(s):` parser; the independently captured baseline confirms 2 sockets × 32 cores and the current parser is corrected. Historical immutable campaign files are preserved.

Source: [Campaign](../validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json) · [Independent audit](../validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md) · [Raw samples](../validation/v2.0.1/campaigns/c20261108_1108/) · [Chart generator](../scripts/make_campaign_charts.py).
=======
| Slide | Chart (assets/slides/) | Core number (campaign c20261108_1108, median of 3 rounds) | Notebook | Raw evidence (validation/v2.0.1/campaigns/c20261108_1108/…) |
|---|---|---|---|---|
| 28–32 | `00_environment.png` | EPYC 9334 ×2 (64C/128T, cgroup quota 16 cores), W7900D gfx1100, ROCm 7.14.1, OpenVINO 2026.4.1 | 00 | `ENV.json` |
| 33 | `01_cpu_vs_gpu.png` | CPU P50 191.3 ms → GPU P50 13.5 ms (**14.14×**), e2e per image; round spread 7.5% / 5.8% | 01 (Lab A) | `labA_cpu_vs_gpu/*.csv` (+summaries + SHA256SUMS) |
| 34–38 | `02_pytorch_vs_openvino_cpu.png` | PyTorch 198.5 ms → OpenVINO 87.2 ms (**2.28×**), same AMD CPU, mAP 0.6398 vs 0.6349 | 02 (Lab B) | `labB_runtime_cpu/*.csv` |
| 39–42 | `03_fp32_vs_int8.png` | FP32 98.3 ms → INT8 78.1 ms (**1.26×**) · 100 MB → 26 MB (**3.8× smaller**) · mAP 0.6349 → 0.5682 (**Δ −0.0667**) · INT8 rebuilt fresh this campaign (identity `867b0ce3…`) | 03 (Lab C) | `labC_int8/*.csv` + `weights/int8_cache/*/CACHE_KEY.json` |
| 43–46 | `04_resolution_vs_latency_accuracy.png` | imgsz 640 → 512: 13.68 → 13.50 ms e2e (−1.3%; fixed pre/post dominates at batch 1) · mAP 0.6398 → 0.6159 (−0.024) | 04 (Lab D-A) | `labD_resolution/*.csv` |
| 47–48 | `05_batch_vs_throughput.png` | batch 1 → 8 forward-only: 95.2 → 144.9 img/s (**1.52×**); 55.2 ms/batch ÷ 8 = 6.9 ms/img-equivalent (**not** request latency); e2e measured separately: 13.6 ms (b1) vs 80.0 ms per 8-image request (b8) | 04 (Lab D-B) | `labD_batch/*batch1_forward*.csv` + `*batch8_forward*.csv` + `*batch1_e2e*.csv` + `*batch8_e2e*.csv` |
| 49–50 | `06_serial_vs_async_pipeline.png` | serial 58.6 → pipeline 83.1 img/s (**1.42×**) on a 100-image burst; per-detection parity 100/100 (min IoU 1.0, max Δconf 0.0 — bit-identical); pipeline e2e p50 611 ms is burst-queue-inclusive (serial e2e 14.0 ms has no queue — different scope) | 05 (Lab E) | `labE_pipeline/*_records.json` + `labE_pipeline_compare_*.json` |
| 51 | all six | "Why faster · how much · at what cost · how proven" | 00–05 | `CAMPAIGN.json` + `AUDIT_REPORT.md` + `SHA256SUMS` |

## The six headline conclusions / 六句话结论（本轮实测）

1. **CPU vs GPU**: same official YOLO26l, batch 1, end-to-end — the W7900 is
   **14.14×** the AMD CPU (P50 13.5 vs 191.3 ms). *Why*: 6144 stream
   processors vs 64 CPU cores for parallel convolutions. *Scope*: this is a
   **hardware/platform comparison**, not a software optimization.
   *Uncertainty*: round spread 7.5% (CPU) / 5.8% (GPU); shared host.
2. **Runtime**: same CPU, same weights — OpenVINO is **2.28×** PyTorch
   (87.2 vs 198.5 ms) at mAP parity (0.6349 vs 0.6398, Δ −0.0049). *Cost*:
   none measured on accuracy; one-time export + first-compile ~1–2 s.
   *Uncertainty*: CPU round spread up to 14.5% (host contention recorded).
3. **INT8**: **1.26×** faster and **3.8×** smaller on the same CPU, but mAP
   drops **−0.0667** with NNCF MIXED PTQ (150 real train2017 calibration
   images, quantization REBUILT this campaign with recorded identity).
   INT8 is NOT accuracy-free — production needs QAT or accuracy-aware flows.
4. **Resolution** 640→512: only **−1.3%** end-to-end latency at batch 1
   (fixed pre/post overhead dominates e2e; forward-only scales more), mAP
   −0.024 — a real speed–accuracy trade-off measured honestly.
5. **Batch** 1→8: +52% GPU forward throughput (95.2 → 144.9 img/s); e2e per
   8-image request 80.0 ms vs 13.6 ms single — **batch time ÷ 8 is an
   equivalent compute cost, never a user latency**.
6. **CPU+GPU pipeline**: **1.42×** burst throughput (58.6 → 83.1 img/s) with
   bit-identical detections (100/100 images, IoU 1.0, Δconf 0.0). Gain is
   bounded (GPU infer ~10 ms already dominates CPU prepare); burst queueing
   RAISES per-request wait (611 ms p50 under a 100-image burst) — the two
   latency scopes are reported separately, never subtracted.

## Fairness & disclosure notes / 公平性与披露

- CPU benchmark variants pin the process to 16 quiet CPUs (one NUMA node,
  re-picked per round, recorded per run) matching the 16-core **cgroup
  quota**; PyTorch CPU threads pinned to 8 (measured optimum under quota).
  Reason: >16 threads triggers cgroup throttling and bimodal latency — see
  `validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.md` (root cause: quota, proven
  via cpu.stat throttle counters).
- **Shared host disclosure**: the machine is shared with other tenants
  (load average 28–42 recorded per round). Numbers are medians of 3 rounds;
  per-round spread is reported in `CAMPAIGN.json → aggregate`.
- GPU timings bracketed by `torch.cuda.synchronize()`; warm-up 10 excluded
  and counted; all raw samples kept (no outlier dropping).
- Lab E: serial and pipeline share ONE predictor decomposition; parity is
  per-DETECTION (class + box IoU ≥ 0.95 + conf ± 0.02, greedy one-to-one);
  input-shape kernels pre-warmed outside the timed window for both modes.
- mAP values are on the *Educational COCO Subset* (120 val2017 images,
  seed=42): comparable within this course, **not an official COCO
  benchmark** — never quote them as COCO leaderboard numbers.
- Verification: `python scripts/verify_evidence.py --campaign
  validation/v2.0.1` (all SHA256SUMS + stats from bytes) and
  `python scripts/audit_benchmarks.py --campaign
  validation/v2.0.1/campaigns/c20261108_1108` (independent recompute).
>>>>>>> b662414 (R2.7 evidence map + R3.8 instructor guide (slides 28-51); README/README.en/results table with campaign c20261108_1108 numbers; 6/6 notebooks executed 45/45 cells)
