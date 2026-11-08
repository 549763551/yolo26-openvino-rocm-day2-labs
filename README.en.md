# YOLO26 Day 02 — AI Inference Acceleration: OpenVINO × AMD ROCm

A hands-on senior-undergraduate course: **one official YOLO26l model, two AMD
compute devices (EPYC CPU + Radeon PRO W7900), six bilingual notebooks, six
real measured experiments — with independently auditable evidence.**

No fabricated numbers anywhere: every figure traces to immutable
per-iteration CSV samples covered by SHA256SUMS and re-computed by an
independent audit script.

---

## Machine & software (v2.0.1 measurement machine)

| | |
|---|---|
| CPU | 2× AMD EPYC 9334 (Zen 4, 64C/128T, 2 NUMA nodes, AVX-512 + VNNI, **no Intel AMX**); container CPU quota 16 cores (reported separately from physical topology) |
| GPU | AMD Radeon PRO W7900D 48 GB (gfx1100, RDNA3); shared host (load recorded) |
| PyTorch | `2.12.0+rocm7.14.1` (AMD ROCm wheel from `repo.amd.com`, version-pinned) |
| OpenVINO | 2026.4.1 CPU plugin (oneDNN) — official validation on AMD CPUs is limited; this repository relies on real-machine measurement |
| Ultralytics | official `yolo26l.pt` (80-class COCO detection, sha256 `9fe3c544…db26`, see `weights/model_identity.json`) |

## The six labs

| # | Notebook | Question |
|---|---|---|
| 00 | [AMD environment & official YOLO26l](notebooks/00_AMD_Environment_and_YOLO26l.ipynb) | Which device & runtime path is it really running on? |
| A | [Lab A: CPU vs GPU baseline](notebooks/01_Lab_A_CPU_vs_GPU_Baseline.ipynb) | Same model — which device wins? Where is the fixed overhead (Amdahl)? |
| B | [Lab B: OpenVINO on the AMD CPU](notebooks/02_Lab_B_OpenVINO_AMD_CPU.ipynb) | Same CPU, different runtime — what changes? |
| C | [Lab C: INT8 quantization](notebooks/03_Lab_C_OpenVINO_INT8.ipynb) | Fewer bits, faster math? At what cost? |
| D | [Lab D: ROCm GPU optimization](notebooks/04_Lab_D_ROCm_GPU_Optimization.ipynb) | Resolution & batch effects on the W7900 |
| E | [Lab E: CPU+GPU async pipeline](notebooks/05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb) | Can CPU+GPU overlap beat serial — and how is overlap proven? |

Every notebook follows one bilingual template: **learning objective → key
idea → predict → run baseline → change ONE thing → compare real data → why →
try it yourself → common mistakes → takeaway**.

## Quick start

```bash
git clone https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs.git
cd yolo26-openvino-rocm-day2-labs
bash scripts/setup.sh                        # one-click install (hardware gates + ROCm torch + data verify/repair + self-checks)
bash scripts/doctor.sh                       # read-only environment diagnostic
bash scripts/run_all.sh --profile quick      # fast in-class measurements
bash scripts/run_all.sh --profile full       # formal benchmark
source .venv/bin/activate && jupyter lab     # open the six notebooks
```

Notebook execution gate (really RUNS all six notebooks, not just structure):

```bash
bash scripts/validate_notebooks.sh             # structure check (CI)
bash scripts/validate_notebooks.sh --execute   # real execution 6/6 + output verification
```

Evidence verification (anyone can re-audit independently):

```bash
python scripts/verify_evidence.py --campaign validation/v2.0.1
python scripts/audit_benchmarks.py --campaign validation/v2.0.1/campaigns/<id>
```

## Measured results (v2.0.1, re-measured on real hardware)

**Campaign `c20261108_1108`** (2026-11-08 · 3 independent rounds × 50 formal
samples per variant · warm-up 10 excluded · median of round P50s · model
sha256 `9fe3c544…db26` · AMD EPYC 9334 + W7900D · shared host with load and
cgroup-throttle counters recorded). Full evidence:
`validation/v2.0.1/campaigns/c20261108_1108/` (CSVs + summaries + SHA256SUMS
+ independent `AUDIT_REPORT.md`).

v2.0.0 historical data remains archived under `results/raw/` (reference only,
no longer used for public claims).

### 1 · Lab A — CPU vs GPU (end-to-end, same official model, batch 1)

| Variant | P50 ms | P95 ms | Throughput img/s |
|---|---|---|---|
| AMD EPYC 9334 (PyTorch, 8 threads, 16-CPU affinity) | 191.3 | 203.1 | 5.2 |
| W7900D (PyTorch ROCm) | 13.5 | 14.2 | 73.9 |

**GPU is 14.14× the CPU** (platform comparison, not a software optimization;
round spread CPU 7.5% / GPU 5.8%)

### 2 · Lab B — same CPU, two runtimes

| Runtime | P50 ms | P95 ms | mAP50-95 |
|---|---|---|---|
| PyTorch CPU | 198.5 | 199.4 | 0.6398 |
| OpenVINO CPU | 87.2 | 99.4 | 0.6349 |

**OpenVINO is 2.28× PyTorch** · ΔmAP −0.0049 (parity) · one-time first compile ~1–2 s

### 3 · Lab C — INT8 quantization (NNCF MIXED, 150 real train2017 calibration images, rebuilt fresh this round)

| Precision | Size MB | P50 ms | mAP50-95 |
|---|---|---|---|
| FP32 | 100 | 98.3 | 0.6349 |
| INT8 | 26 | 78.1 | 0.5682 |

**1.26× faster · 3.8× smaller · ΔmAP −0.0667 (a real accuracy price — never call it lossless)**

### 4 · Lab D-A — resolution (W7900, e2e batch 1)

| imgsz | P50 ms | mAP50-95 |
|---|---|---|
| 640 | 13.68 | 0.6398 |
| 512 | 13.50 | 0.6159 |

Only **−1.3%** end-to-end at batch 1 (fixed pre/post dominates e2e;
forward-only scales more) · ΔmAP −0.024 — a real speed–accuracy trade-off

### 5 · Lab D-B — batch (W7900, imgsz 640)

| Batch | Forward throughput img/s | Latency semantics |
|---|---|---|
| 1 | 95.2 | forward 10.5 ms/req; e2e 13.6 ms/req |
| 8 | 144.9 | forward 55.2 ms/batch; e2e 80.0 ms per 8-image request |

**1.52× forward throughput** · 55.2÷8 = 6.9 ms is an equivalent per-image
compute cost, **not any user's request latency** (both scopes measured and
labeled separately)

### 6 · Lab E — CPU+GPU async pipeline (100-image burst)

| Mode | Throughput img/s | Per-image e2e P50 (scope) |
|---|---|---|
| Serial | 58.6 | 14.0 ms (no queue — pure execution) |
| Async pipeline | 83.1 | 611 ms (burst submit→result, queue included) |

**1.42× throughput** · per-detection parity **100/100 images** (class/box/conf
compared one-to-one, min IoU = 1.0, max Δconf = 0.0 — bit-identical) · the two
latency scopes are reported separately and never subtracted; overlap is proven
by measured stage timelines (Notebook 05 §5b)

*Sources: `validation/v2.0.1/campaigns/c20261108_1108/` (run ids + per-sample
CSVs + SHA256SUMS + `CAMPAIGN.json` + audit report). Charts in `assets/slides/`
are regenerated from this campaign.*

## Teaching arc

1. **Less compute** — input resolution (Lab D-A), INT8 quantization (Lab C)
2. **Faster compute** — OpenVINO on AMD CPU (Lab B), ROCm W7900 (Labs A, D)
3. **Parallel compute** — batching (Lab D-B), async CPU+GPU pipeline (Lab E)

Final goal: students can answer **"why faster · how much · at what cost · how
proven"**.

## Honesty rules

- Only the official `yolo26l.pt` (never a smaller model for nicer numbers).
- "No speedup" or "slower" results are reported and explained (we report what
  we measure).
- Quantization/resolution/batch labs always report the accuracy or resource
  price alongside the speed gain.
- COCO data is an educational subset (*Educational COCO Subset — Not Official
  COCO Benchmark*); manifest + SHA256 in `data/manifest.json`.
- Benchmarks run on a shared host: per-round load and cgroup throttle counters
  are archived with the evidence (CPU runs pinned to 16 quiet CPUs matching
  the 16-core quota).

## Docs

- [Student Guide](docs/STUDENT_GUIDE.md) · [Instructor Guide](docs/INSTRUCTOR_GUIDE.md)
- [Benchmark Protocol](docs/BENCHMARK_PROTOCOL.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Slide 28–51 Evidence Map](docs/SLIDE_28_51_EVIDENCE_MAP.md) · [Data Licenses](docs/DATA_LICENSES.md)
- v2.0.1 validation: `validation/v2.0.1/` (baseline audit / CPU-bimodal
  diagnosis / campaign evidence / FINAL_REPORT.md)

## License

MIT (code & docs). Weights are downloaded at setup time from Ultralytics
(AGPL-3.0 package); COCO images are downloaded at setup time under their
original Flickr licenses (see `docs/DATA_LICENSES.md`). Neither is committed.
