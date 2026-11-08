# Slide 28–51 Evidence Map / 幻灯片证据映射

Every chart below was generated from **real measured data on the lab machine**
(2× AMD EPYC 9334 · Radeon PRO W7900 · torch 2.12.0+rocm7.14.1 · OpenVINO
2026.4.1 · Ultralytics 8.4.207 · official yolo26l.pt, sha256
`9fe3c544…db26`). Each figure embeds its run id and the raw-evidence CSV path
+ SHA256 prefix in its footnote; every number can be re-derived from
`results/raw/` using `course/charts.py`.

Profile: **full** (warmup 10 / 50 timed iterations per latency experiment;
120-image val subset; 150-image calibration subset).

## Slide placement / 图表放置

| Slide | Chart (assets/slides/) | Core number | Notebook | Raw evidence (results/raw/…) |
|---|---|---|---|---|
| 28–32 | `00_environment.png` | EPYC 9334 ×2, W7900 gfx1100, ROCm 7.14.1, OpenVINO 2026.4.1 | 00 | `logs/env_report.json` |
| 33 | `01_cpu_vs_gpu.png` | CPU P50 124.8 ms → GPU P50 13.3 ms (**9.40×**), e2e per image | 01 (Lab A) | `labA_cpu_vs_gpu/labA_cpu_vs_gpu_pytorch_cpu_*.csv` + `…_pytorch_rocm_gpu_*.csv` |
| 34–38 | `02_pytorch_vs_openvino_cpu.png` | PyTorch 154.6 ms → OpenVINO 50.1 ms (**3.09×**), same AMD CPU, mAP 0.6398 vs 0.6349 | 02 (Lab B) | `labB_runtime_cpu/…_pytorch_cpu_*.csv` + `…_openvino_cpu_*.csv` |
| 39–42 | `03_fp32_vs_int8.png` | FP32 49.5 ms → INT8 28.2 ms (**1.75×**) · 100 MB → 26 MB (**3.8× smaller**) · mAP 0.6349 → 0.5682 (Δ −0.067) | 03 (Lab C) | `labC_int8/…_openvino_fp32_*.csv` + `…_openvino_int8_*.csv` |
| 43–46 | `04_resolution_vs_latency_accuracy.png` | imgsz 640 → 512: 13.2 → 12.5 ms e2e (−5%, forward-only scales more) · mAP 0.6398 → 0.6159 (−0.024) | 04 (Lab D-A) | `labD_resolution/…_imgsz640_*.csv` + `…_imgsz512_*.csv` |
| 47–48 | `05_batch_vs_throughput.png` | batch 1 → 8: 97.9 → 145.6 img/s (**1.49×**), VRAM 1.17 GB; 54.9 ms/batch ÷ 8 = 6.9 ms/img equivalent, **not** request latency | 04 (Lab D-B) | `labD_batch/…_batch1_forward_*.csv` + `…_batch8_forward_*.csv` |
| 49–50 | `06_serial_vs_async_pipeline.png` | serial 72.3 → pipeline 87.4 img/s (**1.21×**) on 100 images; detections identical (100/100); pipeline e2e p50 579 ms is burst-queue-inclusive | 05 (Lab E) | `results/latest.json → labE.per_image` (component timing per image) |
| 51 | all six | “Why faster · how much · at what cost · how proven” | 00–05 | `results/RESULTS_TABLE.md` |

## The six headline conclusions / 六句话结论

1. **CPU vs GPU**: same official YOLO26l, batch 1 — W7900 is **9.40×** the AMD
   CPU's speed end-to-end (P50 13.3 ms vs 124.8 ms). *Why*: 6144 stream
   processors vs 64 CPU cores for parallel convolutions.
2. **Runtime**: same CPU, same weights — OpenVINO is **3.09×** PyTorch
   (50.1 vs 154.6 ms) at mAP parity (−0.005). *Cost*: none measured on
   accuracy; export + first-compile is a one-time ~1–2 s.
3. **INT8**: **1.75×** faster and **3.8×** smaller on the same CPU, but mAP
   drops −0.067 with the best PTQ recipe we tested (NNCF MIXED, 150 real
   calibration images; PERFORMANCE preset was worse: −0.106). INT8 is not a
   free lunch — production would need QAT or accuracy-aware pipelines.
4. **Resolution** 640→512: only −5% end-to-end latency at batch 1 (fixed
   pre/post overhead dominates; forward-only scales more), mAP −0.024.
5. **Batch** 1→8: +49% GPU throughput at 1.17 GB VRAM; per-request latency
   semantics change (batch must fill). Equivalent compute cost 6.9 ms/img.
6. **CPU+GPU pipeline**: +21% end-to-end throughput with identical detections;
   the GPU was already the bottleneck (pre ~2 ms vs infer ~10 ms), so the
   overlap gain is honest but bounded — and burst queueing *raises* per-request
   wait time (579 ms p50 under a 100-image burst).

## Fairness notes for the slides / 公平性说明

- PyTorch CPU threads pinned to 8 (measured optimum under this container's
  16-core CPU quota: 8→190 ms, 16→193 ms, 32→297 ms, 64→516 ms);
  thread count recorded in every run summary.
- Ultralytics AutoBackend applies OpenVINO `PERFORMANCE_HINT=LATENCY` for
  batch-1 CPU inference; identical code path for all OpenVINO runs.
- GPU timings bracketed by `torch.cuda.synchronize()`; warm-up runs excluded
  and counted; all raw samples kept (no outlier dropping).
- Lab E: serial and pipeline share one predictor decomposition, verified
  detection-identical to `model.predict` before measurement; all input-shape
  kernels pre-warmed outside the timed window for both modes.
- mAP values are on the *Educational COCO Subset* (120 val2017 images,
  seed=42): comparable within this course, not an official COCO benchmark.
