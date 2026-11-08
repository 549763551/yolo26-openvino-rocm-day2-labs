# Slides 28–51 — v2.0.1 verified evidence map

**Authoritative source:** `validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json`, audited in `AUDIT_REPORT.md`. Branch `release/v2.0.1-r1-r3-validated`. The previous `results/latest.json` and `results/RESULTS_TABLE.md` were v2.0.0 historical records; this file and updated RESULTS_TABLE take precedence for v2.0.1 slides.

**Measurement contract:** 3 independent rounds × 50 timed iterations per latency variant (10 warmups excluded); report the **median of round-level P50s**, not a single-run maximum. Shared Linux machine: 2 × AMD EPYC 9334 (64 physical cores, **16-core cgroup CPU quota**), Radeon PRO W7900D (gfx1100), ROCm PyTorch 2.12.0+rocm7.14.1, OpenVINO 2026.4.1, Ultralytics 8.4.207. Official YOLO26l model SHA256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`.

Every slide should carry the scope (P50 end-to-end vs forward-only vs burst-throughput), source campaign ID, and a brief caveat. **Do not use v2.0.0 numbers as current measurements.**

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
