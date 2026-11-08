# YOLO26 Day 02 — AI Inference Acceleration: OpenVINO × AMD ROCm

A hands-on senior-undergraduate course: **one official YOLO26l model, two AMD
compute devices (2× EPYC 9334 CPU + Radeon PRO W7900 48 GB), six bilingual
notebooks, six real measured experiments.** Every number in this repository
comes from raw per-iteration CSV samples you can audit (`results/raw/`) —
nothing is hand-typed, nothing is fabricated.

## Machine & software

| | |
|---|---|
| CPU | 2× AMD EPYC 9334 (Zen 4, 64C/128T, AVX-512, no AMX) |
| GPU | AMD Radeon PRO W7900 48 GB (gfx1100, RDNA3) |
| PyTorch | `2.12.0+rocm7.14.1` (AMD ROCm wheels from repo.amd.com) |
| OpenVINO | CPU plugin (oneDNN). Note: OpenVINO's official validation on AMD CPUs is limited — everything here is measured on this actual machine |
| Model | official Ultralytics `yolo26l.pt` (80-class COCO detection; SHA256 in `weights/model_identity.json`) |

## The six labs

| # | Notebook | Question |
|---|---|---|
| 00 | [AMD environment & official YOLO26l](notebooks/00_AMD_Environment_and_YOLO26l.ipynb) | Which device is the model really running on? |
| A | [Lab A: CPU vs GPU baseline](notebooks/01_Lab_A_CPU_vs_GPU_Baseline.ipynb) | Same model — which device wins? |
| B | [Lab B: OpenVINO on the AMD CPU](notebooks/02_Lab_B_OpenVINO_AMD_CPU.ipynb) | Same CPU, different runtime — what changes? |
| C | [Lab C: INT8 quantization](notebooks/03_Lab_C_OpenVINO_INT8.ipynb) | Fewer bits, faster math? |
| D | [Lab D: ROCm GPU optimization](notebooks/04_Lab_D_ROCm_GPU_Optimization.ipynb) | How do resolution & batch move GPU speed? |
| E | [Lab E: CPU+GPU async pipeline](notebooks/05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb) | Can heterogeneous overlap beat serial? |

Each notebook: 8–12 core cells following the same template — learn → key idea
→ baseline → change ONE thing → compare real numbers → why → try it yourself
(students modify a parameter) → takeaway. All explanations are bilingual
(English / 中文).

## Quick start

```bash
git clone https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs.git
cd yolo26-openvino-rocm-day2-labs
bash scripts/setup.sh                     # one-click install + self-test
bash scripts/run_all.sh --profile quick   # fast in-class measurements
bash scripts/run_all.sh --profile full    # formal citable benchmark
source .venv/bin/activate && jupyter lab  # open the six notebooks
```

Diagnostics: `bash scripts/doctor.sh` · restore on cloud VMs:
`bash scripts/restore_env.sh` · tests: `python -m pytest -q`.

## Measured results

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

*Source: `results/latest.json` + `results/raw/` (per-iteration latency CSV with
SHA256 evidence chains). Charts in `assets/slides/` are regenerated from raw
data by `course/charts.py`.*

## Teaching arc

1. **Less compute** — input resolution (Lab D-A), INT8 quantization (Lab C)
2. **Faster compute** — OpenVINO on the AMD CPU (Lab B), ROCm W7900 (Labs A, D)
3. **Parallel compute** — batching (Lab D-B), async CPU/GPU pipeline (Lab E)

Final exam question of the day: *why faster, how much faster, at what cost,
and how do you prove it?*

## Honesty rules

- Only the official `yolo26l.pt` — never a smaller model for better numbers.
- "No speedup" or "slower" are valid results and are explained, not hidden.
- Quantization/resolution/batch experiments always report the accuracy or
  resource cost alongside the speedup.
- COCO data is an educational subset (*Educational COCO Subset — Not Official
  COCO Benchmark*); manifest with ids/SHA256/licenses: `data/manifest.json`.

## Docs

- [Student guide](docs/STUDENT_GUIDE.md) · [Instructor guide](docs/INSTRUCTOR_GUIDE.md)
- [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Slide 28–51 evidence map](docs/SLIDE_28_51_EVIDENCE_MAP.md) · [Data licenses](docs/DATA_LICENSES.md)

## License

MIT (code & docs). Weights (Ultralytics, AGPL-3.0 package) and COCO images
(original Flickr licenses) are downloaded at setup time, never committed.
