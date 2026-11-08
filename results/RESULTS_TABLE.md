# YOLO26 Day 02 — Verified Results Table / 核验结果表

**Version:** v2.0.1 campaign `c20261108_1108` (documentation correction for next release, **not a new GPU run**).

**Source of truth:** [CAMPAIGN.json](../validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json) · [53-run independent CSV audit](../validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md) · [SHA256SUMS](../validation/v2.0.1/campaigns/c20261108_1108/SHA256SUMS). All performance results are from real AMD EPYC 9334 + W7900D (gfx1100), 3 rounds × 50 timed samples, 10 warmups excluded, **median of per-round P50/P95**. Shared-host + 16-core container quota; 120-image educational COCO subset, **not** official COCO leaderboard.

## Corrected tail latencies / 修正后的 P95

| Variant | P50 ms | **P95 ms** | Measurement scope |
|---|---:|---:|---|
| Lab A · AMD CPU / PyTorch | 191.28 | **470.63** | per-image end-to-end |
| Lab A · W7900D / ROCm | 13.53 | **15.73** | per-image end-to-end |
| Lab B · AMD CPU / PyTorch | 198.54 | **441.02** | per-image end-to-end |
| Lab B · AMD CPU / OpenVINO | 87.22 | **93.85** | per-image end-to-end |
| Lab C · OpenVINO FP32 | 98.30 | **102.13** | per-image end-to-end |
| Lab C · OpenVINO INT8 | 78.14 | **83.43** | per-image end-to-end |
| Lab D-A · W7900D, imgsz 640 | 13.68 | **14.96** | per-image end-to-end |
| Lab D-A · W7900D, imgsz 512 | 13.50 | **20.04** | per-image end-to-end |
| Lab D-B · W7900D, batch=1 forward | 10.51 | **11.77** | forward-only per image |
| Lab D-B · W7900D, batch=8 forward | 55.22 | **56.02** | forward-only per 8-image batch |
| Lab D-B · W7900D, batch=1 E2E | 13.58 | **14.86** | per 1-image request |
| Lab D-B · W7900D, batch=8 E2E | 80.04 | **83.29** | per 8-image request |
| Lab E · serial | 13.86 | **16.54** | per-image execution, no queue |
| Lab E · async pipeline | 603.54 | **1133.92** | 100-image burst queue-inclusive latency |

**Previous defect:** published v2.0.1 README / English README used `p50_max_ms` for some P95 fields. Above values instead use `aggregate.<variant>.p95_median_ms`. The original data files were never altered. In particular, Lab A CPU and Lab B CPU have severe tail latency that must be taught transparently.

## Six real-hardware results / 六组实测结论

| Experiment | Baseline → candidate | Result from P50 or wall throughput | Accuracy / limitations |
|---|---|---|---|
| Lab A CPU vs GPU | 191.28→13.53 ms | **14.14×** lower median latency | Hardware/platform comparison, **not software speedup** |
| Lab B CPU runtime | 198.54→87.22 ms | **2.28×** lower median latency | mAP 0.6398→0.6349 (Δ−0.0049); P95 441.02→93.85 |
| Lab C INT8 | 98.30→78.14 ms | **1.26×** lower median latency | mAP 0.6349→0.5682 (Δ−0.0667), **not lossless** |
| Lab D-A resolution | 640→512: 13.68→13.50 ms | ≈**1.3%** lower median E2E latency | mAP 0.6398→0.6159 (Δ−0.0239), **P95 worsens** |
| Lab D-B batch | forward-only 95.2→144.9 images/s | **1.52×** forward-equivalent throughput | Batch completion 55.22 ms; `÷8` ≠ user latency; E2E b8 80.04 ms/batch |
| Lab E async pipeline | serial 58.6→async 83.1 images/s on 100-image burst | **1.42×** burst throughput | 100/100 detections matched, shared host; no comparable single-request latency improvement claimed |

### Model sizes / 模型文件大小

**v2.0.1 fresh model-size measurement: NOT RECORDED.** The frequently cited ≈100 MB FP32 and ≈26 MB INT8 (≈3.8× smaller) come from **v2.0.0 historical physical exports**. They must not be represented as a freshly measured v2.0.1 result. To obtain new evidence, use `python scripts/measure_openvino_sizes.py` on real, freshly exported OpenVINO FP32/INT8 XML/BIN files. This is an **AMD CPU/local filesystem task; W7900 GPU benchmarking is not required**.

### Reproduce / 审核与复现

```bash
python scripts/verify_evidence.py --campaign validation/v2.0.1
python scripts/audit_benchmarks.py --campaign validation/v2.0.1/campaigns/c20261108_1108
python -m pytest -q tests/test_release_docs.py
```

The v2.0.0 results remain archived at `results/raw/`. Existing `results/latest.json` is historical and **not** the v2.0.1 authoritative campaign. Never combine historical and new runs.
