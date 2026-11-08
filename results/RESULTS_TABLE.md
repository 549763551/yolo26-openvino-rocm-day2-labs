<<<<<<< HEAD
# YOLO26 Day 02 — authoritative v2.0.1 campaign results

**Campaign:** `c20261108_1108` · 2026-11-08 · official YOLO26l (COCO 80-class detection).
**Scope:** 3 independent rounds × 50 timed iterations per latency variant, 10 warmups excluded. Summary uses the **median of each round's P50**, not pooling raw samples across runs. Machine: AMD EPYC 9334 ×2 (cgroup 16-core quota), Radeon PRO W7900D (gfx1100), ROCm PyTorch 2.12.0+rocm7.14.1, OpenVINO 2026.4.1. All speedups are specific to this shared host.

| Lab | Comparison | Baseline P50 | Candidate P50 | Result |
|---|---|---:|---:|---:|
| A | EPYC PyTorch CPU → W7900 ROCm GPU; per-image E2E | 191.28ms | 13.53ms | **14.14×** |
| B | EPYC PyTorch CPU → OpenVINO CPU; per-image E2E | 198.54ms | 87.22ms | **2.28×** |
| C | OpenVINO FP32 CPU → MIXED INT8 CPU; per-image E2E | 98.30ms | 78.14ms | **1.26×** |
| D-A | imgsz 640 → 512; W7900 per-image E2E | 13.68ms | 13.50ms | **1.013×** (≈1.3% lower latency) |
| D-B | batch 1 → 8, **forward-only** | 95.2 img/s | 144.9 img/s | **1.52×** throughput |
| E | serial → async CPU+GPU; 100-image burst workload | 58.62 img/s | 83.13 img/s | **1.42×** throughput |

### Accuracy, latency semantics and cautions

| Test | Baseline | Candidate | Correct interpretation |
|---|---:|---:|---|
| Lab B mAP50–95 | 0.6398 | 0.6349 | −0.0049 on 120-image educational subset |
| Lab C mAP50–95 | 0.6349 | 0.5682 | **ΔmAP = −0.0667**, important accuracy cost |
| Lab D-A mAP50–95 | 0.6398 | 0.6159 | ΔmAP = −0.0239; input 640²→512² |
| Lab D-A P95 | 14.96ms | 20.04ms | P50 improvement is small and P95 worsens in this run aggregate |
| Lab D-B forward time | 10.51ms / image (b1) | 55.22ms / batch (b8) | 6.90ms/image equivalent at b8 is **not** request latency |
| Lab D-B E2E | 13.58ms (b1) | 80.04ms per 8-image batch | Different batch request sizes; completion time increases |
| Lab E request P50 | 13.96ms (serial, no queue) | 610.81ms (full-burst pipeline, includes queue) | **Different arrival/queue semantics**; do not call pipeline lower latency |

**Not measured by this campaign:** resolution/batch GPU allocator VRAM; do not show 0.00GB. Archived v2.0.0 exported model sizes ≈100MB/26MB must be labelled as historical if displayed.

**Provenance:** [Campaign JSON](../validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json) · [Independent 53-run audit](../validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md) · [96-file SHA256SUMS](../validation/v2.0.1/campaigns/c20261108_1108/SHA256SUMS) · [Slides map](../docs/SLIDE_28_51_EVIDENCE_MAP.md).

Original v2.0.0 results remain in Git history and `results/latest.json` for historical comparison only.
=======
# Results Table / 结果汇总

## v2.0.1 — Campaign `c20261108_1108` (current, citable)

3 independent rounds × 50 formal samples per latency variant (warm-up 10
excluded), median of round P50s; model sha256 `9fe3c544…db26`; 2× AMD EPYC
9334 + Radeon PRO W7900D; ROCm PyTorch 2.12.0+rocm7.14.1 · OpenVINO 2026.4.1 ·
NNCF 3.4.0 · Ultralytics 8.4.207. Shared host — per-round load and cgroup
throttle counters recorded in `CAMPAIGN.json`. Evidence:
`validation/v2.0.1/campaigns/c20261108_1108/` · audit:
`validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md` (53 runs
recomputed independently, PASS).

| Lab | Variant | P50 ms | P95 ms | Extra |
|---|---|---|---|---|
| A | AMD CPU (PyTorch, 8 thr, 16-CPU affinity) | 191.3 | 203.1 | 5.2 img/s |
| A | W7900D (PyTorch ROCm) | 13.5 | 14.2 | 73.9 img/s |
| B | PyTorch CPU | 198.5 | 199.4 | mAP .6398 |
| B | OpenVINO CPU | 87.2 | 99.4 | mAP .6349 |
| C | OpenVINO FP32 | 98.3 | 98.9 | 100 MB · mAP .6349 |
| C | OpenVINO INT8 | 78.1 | 79.2 | 26 MB · mAP .5682 |
| D-A | imgsz 640 (e2e) | 13.68 | — | mAP .6398 |
| D-A | imgsz 512 (e2e) | 13.50 | — | mAP .6159 |
| D-B | batch 1 (forward) | 10.5 | — | 95.2 img/s · e2e 13.6 ms/req |
| D-B | batch 8 (forward) | 55.2/batch | — | 144.9 img/s · e2e 80.0 ms/8-img request |
| E | serial | — | — | 58.6 img/s · e2e P50 14.0 ms (no queue) |
| E | async pipeline | — | — | 83.1 img/s · e2e P50 611 ms (burst, queue incl.) |

**Headline ratios (median-of-rounds P50):**

| # | Conclusion | Value | Cost / scope |
|---|---|---|---|
| 1 | W7900 vs EPYC (same model, e2e) | **14.14×** | platform comparison; spread ≤ 7.5% |
| 2 | OpenVINO vs PyTorch (same CPU) | **2.28×** | mAP parity (−0.0049); one-time compile |
| 3 | INT8 vs FP32 | **1.26×** · 3.8× smaller | **ΔmAP −0.0667** (not lossless) |
| 4 | imgsz 640→512 | −1.3% e2e | mAP −0.024 |
| 5 | batch 1→8 (forward) | **1.52×** throughput | 55.2÷8 = 6.9 ms/img-equivalent ≠ request latency |
| 6 | async CPU+GPU pipeline | **1.42×** burst throughput | detections bit-identical (100/100, IoU 1.0); burst queue raises per-request wait |

INT8 identity: rebuilt fresh in this campaign — NNCF MIXED, 150 real
train2017 calibration images, quantization time 44 s, identity key recorded in
`weights/int8_cache/*/CACHE_KEY.json` (calibration dataset sha `867b0ce3…`).

---

## v2.0.0 — historical (archived, superseded; reference only)

Measured 2026-11-08 morning, single full pass (warm-up 10 / 50 iters), code at
tag `v2.0.0` (commit `e7b684b`). Raw samples: `results/raw/` (note: v2.0.0
summary JSONs carry the self-hash defect B-02 — their `json_sha256` fields do
not match file bytes; CSVs remain valid measurements).

| # | Conclusion (v2.0.0) | Value |
|---|---|---|
| 1 | GPU vs CPU | 9.40× (124.8 → 13.3 ms) |
| 2 | OpenVINO vs PyTorch CPU | 3.09× (154.6 → 50.1 ms) |
| 3 | INT8 vs FP32 | 1.75× · ΔmAP −0.0667 |
| 4 | resolution 640→512 | −5% e2e · ΔmAP −0.024 |
| 5 | batch 1→8 | 1.49× (97.9 → 145.6 img/s) |
| 6 | pipeline | 1.21× (72.3 → 87.4 img/s) |

Differences vs v2.0.1 are attributable to (a) shared-host load (recorded per
round in v2.0.1 — loadavg 28–42 during the campaign), (b) v2.0.1 CPU runs
pinned to 16 quiet CPUs matching the 16-core cgroup quota (removing quota
throttling; see `validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.md`), and (c)
v2.0.1's multi-round median protocol. Both datasets are preserved; only the
v2.0.1 campaign is citable for current claims.
>>>>>>> b662414 (R2.7 evidence map + R3.8 instructor guide (slides 28-51); README/README.en/results table with campaign c20261108_1108 numbers; 6/6 notebooks executed 45/45 cells)
