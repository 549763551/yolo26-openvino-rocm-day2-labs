### Six measured results / 六组实测结果

profile: **full** · generated: 2026-11-08T09:56:56 · model: official yolo26l.pt (80-class COCO detection)

All values measured on this machine (2× AMD EPYC 9334 + Radeon PRO W7900, ROCm PyTorch 2.12.0+rocm7.14.1). Raw samples: `results/raw/`.

| Lab | Device | P50 ms | P95 ms | Throughput img/s |
|---|---|---|---|---|
| A baseline | AMD CPU (PyTorch) | 124.8 | 160.0 | 8.0 |
| A baseline | W7900 (PyTorch ROCm) | 13.3 | 26.1 | 75.3 |

GPU vs CPU P50 speedup: **9.40x**

| Lab | Runtime (same AMD CPU) | P50 ms | P95 ms | mAP50-95 |
|---|---|---|---|---|
| B runtime | PyTorch | 154.6 | 183.6 | 0.6398 |
| B runtime | OpenVINO | 50.1 | 75.4 | 0.6349 |

OpenVINO vs PyTorch P50: **3.09x** · ΔmAP: -0.0049

| Lab | Precision | Size MB | P50 ms | mAP50-95 |
|---|---|---|---|---|
| C quant | FP32 | 100 | 49.5 | 0.6349 |
| C quant | INT8 | 26 | 28.2 | 0.5682 |

INT8 vs FP32 P50: **1.75x** · size 3.79x smaller · ΔmAP: -0.0667

| Lab | Config | P50 ms | mAP50-95 | VRAM GB |
|---|---|---|---|---|
| D-A resolution | imgsz 640 | 13.2 | 0.6398 | 0.36 |
| D-A resolution | imgsz 512 | 12.5 | 0.6159 | 0.33 |

640→512 latency: **1.06x faster** · ΔmAP: -0.0239

| Lab | Batch | Throughput img/s | Latency semantics |
|---|---|---|---|
| D-B batch | 1 | 97.9 | per-request 10.2 ms |
| D-B batch | 8 | 145.6 | batch 54.9 ms (equiv 6.9 ms/img, NOT request latency) |

Batch throughput gain: **1.49x** · VRAM peak 1.17 GB

| Lab | Mode | Total s | img/s | e2e P50 ms |
|---|---|---|---|---|
| E pipeline | serial | 1.4 | 72.3 | 13 |
| E pipeline | async CPU+GPU | 1.1 | 87.4 | 579 |

Pipeline throughput gain: **1.21x** · detection consistency: PASS (100 images)

