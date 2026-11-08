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
