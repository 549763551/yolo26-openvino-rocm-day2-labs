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

## Measured results (v2.0.1 · 2026-11-08)

The authoritative measurements for this branch are in the **[v2.0.1 physical-hardware campaign](validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json)**: three independent rounds, 50 timed iterations per latency variant, median of each round's P50. The older `results/latest.json` is a **v2.0.0 historical snapshot**, not the current slide data.

| Lab | Comparison | v2.0.1 measurement | Accuracy/latency caveat |
|---|---|---:|---|
| A | EPYC CPU → W7900 ROCm GPU | **14.14×** (P50 191.28→13.53ms) | Hardware comparison, per-image E2E |
| B | PyTorch CPU → OpenVINO CPU | **2.28×** (198.54→87.22ms) | Same AMD CPU, mAP −0.0049 |
| C | OpenVINO FP32 → INT8 | **1.26×** (98.30→78.14ms) | **mAP −0.0667** |
| D-A | Input 640 → 512 | **≈1.3%** lower P50 (13.68→13.50ms) | mAP −0.0239; P95 worsens |
| D-B | Batch 1 → 8, forward-only | **1.52×** throughput (95.2→144.9 img/s) | Batch completion time is NOT request latency |
| E | Serial → async CPU+GPU | **1.42×** throughput (58.62→83.13 img/s) | Burst queue-inclusive P50 ≈611ms |

**Scope:** 2× AMD EPYC 9334 + Radeon PRO W7900D (gfx1100), with a **16-core cgroup CPU quota**; accuracy on a 120-image educational COCO subset (not official COCO). Resolution/batch VRAM not measured in v2.0.1; missing is not zero. Archived 100/26MB FP32/INT8 export sizes are from v2.0.0, if referenced.

- [Full v2.0.1 results table](results/RESULTS_TABLE.md)
- [Slide 28–51 evidence mapping](docs/SLIDE_28_51_EVIDENCE_MAP.md)
- [Independent campaign audit: 53 runs, 96 checksums](validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md)

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
