# Instructor Guide / 教师指南

Target audience: senior undergraduates (28 students, 5 groups).
Machine: 2× AMD EPYC 9334 (64C/128T) + Radeon PRO W7900 48 GB (gfx1100),
ROCm PyTorch 2.12.0+rocm7.14.1, OpenVINO, NNCF, Ultralytics.

## Pre-class checklist / 课前准备（关键！）

Run this **before** class (downloads + exports + quantization are cached):

```bash
bash scripts/setup.sh                       # idempotent; ~15 min first time
bash scripts/run_all.sh --profile full      # pre-generates all evidence + charts (~30-60 min)
bash scripts/doctor.sh                      # verify all gates pass
```

Why pre-run: the OpenVINO export and the INT8 quantization (NNCF calibration on
150 images) take several minutes each on the CPU. Notebooks detect the cached
artifacts and become near-instant, so class time is spent on *measurements and
reasoning*, not waiting. This is explicitly allowed by the course rules —
quick mode reduces repetitions, never the model.

## Timing table (measured on the lab machine) / 课堂时间表（真机实测）

Notebook wall-clock measured headlessly (nbconvert, quick defaults, cached
exports/quantization — see `validation/linux_w7900/notebook_timings/`):

| Notebook | Executed wall time | In-class budget (incl. discussion) |
|---|---|---|
| 00 Environment & YOLO26l | 9 s | 10 min |
| A CPU vs GPU | 14 s | 15 min |
| B OpenVINO CPU | 45 s | 20 min |
| C INT8 | 31 s (quantization cached; first quantize ≈ 5 min) | 20–30 min |
| D GPU res/batch | 32 s | 20 min |
| E CPU+GPU pipeline | 12 s | 20 min |

Compute is never the bottleneck — spend class time on predictions,
measurements and the "why" discussions. The `full` profile benchmark
(50 iterations, all charts) took ~5 minutes headlessly.

## Teaching arc (Slides 28–51) / 教学主线

1. **Less compute** (input size, INT8) — Labs C, D-A.
2. **Faster compute** (OpenVINO CPU, ROCm GPU) — Labs B, A, D.
3. **Parallel compute** (batch, async pipeline, CPU+GPU) — Labs D-B, E.

The final exam question of the day: *why faster, how much faster, at what
cost, and how do you prove it?* / 为什么快、快多少、代价是什么、如何证明。

## Where the evidence lives / 证据位置

- Raw per-iteration samples: `results/raw/<experiment>/*.csv`
- Run summaries (env, params, stats, sha256): `results/raw/<experiment>/*_summary.json`
- Merged view: `results/latest.json`
- Slide charts: `assets/slides/*.png|.svg` (regenerate from CSV/JSON —
  `python scripts/run_experiments.py --profile full` or chart functions in
  `course/charts.py`).
- Slide mapping: `docs/SLIDE_28_51_EVIDENCE_MAP.md`

## Fair-comparison rules to enforce in class / 课堂公平对比规则

- Same official weights (sha256 in `weights/model_identity.json`).
- Same images (`data/val_subset`, fixed manifest).
- Change ONE variable at a time; record warm-up and iteration counts.
- Batch throughput ≠ per-request latency — make students say which one they
  measured and why.
- If a result disappoints (no speedup), that is a *teaching moment*: use the
  component timings (pre/infer/post) to find the bottleneck.

## Common student errors / 常见问题

- Timing GPU calls without `torch.cuda.synchronize()` → "0.3 ms" fantasy
  numbers. Our `bench()` helper does this for them; ask them why it matters.
- Comparing `imgsz=640` mAP with `imgsz=512` mAP computed on different images.
- Dividing batch time by 8 and calling it latency (Lab D-B trap).
- Forgetting that pre/post-processing still runs on the CPU even when the
  model runs on the GPU (why Lab A gaps are not extreme at batch 1).
