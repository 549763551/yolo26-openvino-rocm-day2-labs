# Slide 28–51 Evidence Map / 幻灯片证据映射（v2.0.1 corrected）

> **Authoritative lesson plan / 授课基准：** 已确认的 Day 02 六模块、24 页课程大纲。每页编号固定；本文件替代 v2.0.1 合并冲突产生的两套相互矛盾页码。**This documentation correction does not create new benchmark measurements.**

**Measured evidence / 实测依据：** [campaign c20261108_1108](../validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json) (3 rounds × 50 timed iterations per latency variant, 10 warmups excluded; median of per-round P50/P95). **Model:** official YOLO26l only, SHA256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`. **Hardware:** 2× AMD EPYC 9334 (16-core container quota), Radeon PRO W7900D gfx1100, ROCm PyTorch 2.12.0+rocm7.14.1, OpenVINO 2026.4.1. Shared host; values are not hardware-independent guarantees. **Accuracy:** 120-image educational COCO subset, not official COCO leaderboard metrics.

| Slide | 已确认的正式标题 / Fixed teaching topic | 对应实验 / Evidence | 素材与使用边界 / Visual and caveats |
|---|---|---|---|
| 28 | 模型训练完成，为什么还需要推理优化？ | Official pretrained YOLO26l used throughout Day 02 | `00_environment.png`, conceptual; do **not** imply Day 01 fine-tuned weights used |
| 29 | 一张图片经过哪些步骤才能得到 YOLO 检测结果？ | Decode → preprocess → transfer → infer → postprocess → result | Conceptual pipeline; stage boundaries must be explicit |
| 30 | 推理时间到底花在哪里？ | Lab A model-only vs end-to-end + Amdahl example | Notebook 01, illustrative 30→25 ms is **not** measured W7900 data |
| 31 | Latency、FPS 和 Throughput 有什么区别？ | Lab D Batch and Lab E latency semantics | Measured scope labels; Batch latency ≠ per-image equivalent cost |
| 32 | 如何设计一次公平的 AI Benchmark？ | `CAMPAIGN.json`, `ENV.json`, `SHA256SUMS` | Warmup=10; 3 rounds × 50; P50/P95; shared-host limits |
| 33 | Lab A：建立 YOLO26 的第一份性能基线 | Lab A: AMD CPU P50 191.28 ms / **P95 470.63 ms**; W7900D P50 13.53 ms / **P95 15.73 ms** | `01_cpu_vs_gpu.png`; P50 platform ratio 14.14×, *not* software-only acceleration |
| 34 | 模型训练格式与部署格式有什么区别？ | YOLO26l `.pt` → OpenVINO IR | Notebook 02, conceptual export diagram |
| 35 | OpenVINO 如何在 AMD CPU 上执行神经网络？ | `ov.Core()` → `read_model` → `compile_model("CPU")` → infer | Notebook 02 runtime demo; AMD CPU is measured, not vendor-certified |
| 36 | 如何把 YOLO26 导出为 OpenVINO 模型？ | Same official model SHA, OpenVINO export | Notebook 02; XML/BIN and model identity |
| 37 | 为什么 OpenVINO 可能加速 AMD CPU 推理？ | Lab B CPU PyTorch P50 198.54 / **P95 441.02** vs OpenVINO P50 87.22 / **P95 93.85** ms | `02_pytorch_vs_openvino_cpu.png`; P50 ratio 2.28×; disclose CPU quota and cross-round variation |
| 38 | Lab B：PyTorch CPU vs OpenVINO AMD CPU | Lab B mAP 0.6398→0.6349, Δ−0.0049 | Same CPU, workload, weights; 120-image educational subset |
| 39 | FP32、FP16 和 INT8 究竟有什么区别？ | Quantization concept and numeric examples | Notebook 03 conceptual, no unsupported claim of AMD Intel AMX |
| 40 | INT8 为什么可能在 AMD CPU 上更快、更省内存？ | Lab C P50 98.30→78.14 ms, P95 102.13→83.43 ms | `03_fp32_vs_int8.png`; 1.26× P50, **model size figures in this chart originate from v2.0.0 and are historical, NOT remeasured in v2.0.1** |
| 41 | 模型压缩后，检测准确性会变化吗？ | mAP50–95 0.6349→0.5682 (Δ−0.0667) | Accuracy tradeoff; don't claim lossless INT8 |
| 42 | Lab C：OpenVINO AMD CPU INT8 量化 | NNCF MIXED PTQ, 150 real train2017 calibration images, cache identity | Notebook 03; size needs **new local export measurement** before new-release claim |
| 43 | 为什么 GPU 适合深度学习推理？ | Lab A GPU model-only vs CPU and GPU path architecture | Conceptual hardware parallelism, avoid simple core-count argument |
| 44 | YOLO26 如何通过 ROCm 在 AMD GPU 上运行？ | PyTorch ROCm/HIP → W7900D, gfx1100 | Notebook 00/04 hardware verification |
| 45 | 输入分辨率越小，推理一定越好吗？ | Lab D-A 640→512: P50 13.68→13.50 ms; mAP 0.6398→0.6159 | `04_resolution_vs_latency_accuracy.png`; small ≈1.3% P50 shift, not necessarily significant; P95 14.96→20.04 ms |
| 46 | Batch=1 与 Batch=8，哪个更快？ | Lab D-B: forward-only throughput 95.2→144.9 img/s | `05_batch_vs_throughput.png`; forward-only 1.52×, e2e b1 13.58 ms vs b8 80.04 ms/batch |
| 47 | GPU 算得很快，为什么整个系统仍然慢？ | Lab A/D stage timing, transfer/sync/CPU overhead | Amdahl and pre/post boundary; no synthetic speedup |
| 48 | Lab D：AMD ROCm GPU 推理优化 | Batch and resolution A/B, accuracy, raw CSV | Notebook 04; no v2.0.1 allocator VRAM data (do not show 0 GB) |
| 49 | AMD CPU 和 GPU 怎样分工，才可能更高效？ | Lab E real serial/async stage records, bounded queue | `06_serial_vs_async_pipeline.png`; distinguish actual overlap vs queueing |
| 50 | Lab E：串行推理 vs CPU/GPU 异步流水线 | 100-image burst: 58.6→83.1 img/s (1.42×); 100/100 matched detections | Queue-inclusive async P50 ≈603.5 ms vs serial execution P50 ≈13.9 ms: **different arrival scenarios**; do not subtract |
| 51 | 怎样证明你的 AI 推理优化真的有效？ | Six complete labs, correctness + tradeoff + evidence seal | `CAMPAIGN.json`, `AUDIT_REPORT.md`, raw CSV, SHA256SUMS |

## Statistical correction / 统计更正

`P50` and `P95` in this map always mean **median of the three round-level percentiles** (the same definition as `aggregate.<key>.p50_median_ms` and `p95_median_ms`). Do **not** use `p50_max_ms` as P95. The former README mixed those statistics. For CPU tail latency, the difference is material (Lab A CPU true P95 ≈470.6 ms rather than the incorrectly displayed 203.1 ms). [Raw run audit](../validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md).

## Model size status / 模型文件大小的证据状态

The earlier FP32 ≈100 MB / INT8 ≈26 MB figures were physically measured during **v2.0.0**. They are **not v2.0.1 campaign data**. Size bars on existing charts must carry that historical label, and new release tables must not claim these as freshly measured without re-reading the actual XML/BIN from the new export and recording hashes. The [measurement tool](../scripts/measure_openvino_sizes.py) is provided; fresh size measurements are **PENDING** until executed where the exported models live.

## Caveats / 不可越界的结论

- CPU/GPU ratio is a **platform comparison**.
- Shared EPYC host had **16-core cgroup quota** and neighbor contention; these timings aren't universal.
- INT8 mAP decline (−0.0667) is significant; the educational COCO subset is **not official COCO mAP**.
- Forward-only Batch throughput and full-image throughput are different.
- Lab E serial and burst pipeline per-request latencies have different queue scopes, so compare *throughput*, not those request P50s.
- The campaign contains an earlier CPU topology parsing defect in `ENV.json` (zeros); corrected topology was independently recorded elsewhere, and the sealed historical evidence is intentionally unchanged.
- Do not modify original `CAMPAIGN.json`, `SHA256SUMS` or raw CSV to “fix” presentation errors.
