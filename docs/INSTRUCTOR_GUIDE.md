# Instructor Guide — YOLO26 Day 02 (v2.0.1)

Slides 28–51, taught with six notebooks. All performance numbers below come
from campaign `validation/v2.0.1/campaigns/c20261108_1108` (3 rounds × 50
samples, median of round P50s, independently audited). Machine: 2× AMD EPYC
9334 (64C/128T, container quota 16 cores) + Radeon PRO W7900D (gfx1100) +
ROCm PyTorch 2.12.0+rocm7.14.1 + OpenVINO 2026.4.1 · official yolo26l.pt
(sha256 `9fe3c544…db26`).

Total suggested time: 90 min (30 min lecture + 45 min notebooks + 15 min
discussion). Notebook quick-mode wall times (measured this round):
00: 7 s · 01: 13 s · 02: 56 s · 03: 30 s (first-ever INT8 quantize ≈ +44 s) ·
04: 26 s · 05: 9 s.

---

## Slide 28–32 · Environment & two execution paths — Notebook 00

**Core knowledge**: before any performance claim, establish (1) hardware
identity (AMD EPYC 9334 ×2, W7900 gfx1100), (2) software identity (ROCm
torch, OpenVINO, Ultralytics versions), (3) model identity (official
yolo26l.pt, sha256, 80 classes, detect task).

**Draw on the board** (also in NB00): the same yolo26l.pt has TWO runtime
paths — OpenVINO CPU plugin (oneDNN) on the EPYC, and PyTorch+ROCm/HIP on
the W7900. **OpenVINO does not touch the GPU in this course.**

**Interaction**: "Which of the two paths executes Lab B? Lab D?" (Answer: B/C
= OpenVINO-CPU; A/D/E = ROCm-GPU.)

**Common misconception** ❌: "OpenVINO is driving the W7900." Correct ✅:
OpenVINO runs the AMD CPU path; the GPU path is PyTorch/ROCm.
❌: "torch.cuda means NVIDIA." Under ROCm the same API targets AMD GPUs.

**Numbers you may quote**: EPYC 9334: 64 physical cores / 128 threads, 2 NUMA
nodes, AVX-512 + VNNI, **no Intel AMX**; W7900: gfx1100, 48 GB; container CPU
quota = 16 cores (quota ≠ physical topology — this becomes important in Lab B).

**Never say**: "AMD EPYC has AMX."

---

## Slide 30 · Fixed overhead & Amdahl — Notebook 01 (mini-lab)

**Core knowledge**: end-to-end latency = CPU pre/post (fixed) + model compute.
Speeding up only the model part has a bounded system effect.

**Board example (students compute by hand first)**: 20 ms CPU + 10 ms GPU =
30 ms. GPU 2× faster → 20 + 5 = **25 ms**, not 15 ms. System speedup 1.2×
despite 2× on the GPU part. Limit with an infinitely fast GPU: 20 ms (1.5×).

**Real data**: measured GPU e2e P50 ≈ 13.5 ms vs model-only forward ≈ 10.5 ms
→ ≈ 3 ms fixed non-model overhead (pre/post + transfers) — matches the
mini-lab's lesson.

**Interaction**: "If the CPU part were 90% of the time, what's the max
speedup from a 10× faster GPU?" (1.11×.)

**Never say**: "GPU X times faster ⇒ the system is X times faster."

---

## Slide 33 · CPU vs GPU baseline — Notebook 01 (Lab A)

**Core knowledge**: latency (P50/P95), throughput, warm-up; e2e scope
(pre + infer + post); GPU async timing needs synchronization.

**Measured (campaign c20261108_1108)**: CPU P50 191.3 ms (P95 ~203) vs W7900
P50 13.5 ms (P95 ~14.2) → **14.14×** end-to-end, same official model, batch 1,
imgsz 640.

**Teaching points**: this is a **platform comparison** (different hardware),
NOT a software optimization — that framing matters for slides 34+.
Round-to-round spread: CPU 7.5%, GPU 5.8% (shared host, disclosed).

**Interaction**: students predict the ratio before running (typical guesses:
2–5×). Then discuss why the truth is ~14×: 6144 stream processors vs 64 CPU
cores for convolutions, but fixed CPU work caps it (Amdahl again).

**Common mistake** ❌: timing GPU work without `torch.cuda.synchronize()`.

---

## Slide 34–38 · Same CPU, two runtimes — Notebook 02 (Lab B)

**Core knowledge**: an inference runtime = graph optimization + kernel
selection + threading. Export once (IR), compile per target, then infer.

**Measured**: PyTorch CPU 198.5 ms → OpenVINO CPU 87.2 ms P50 = **2.28×**
(same CPU, same weights, same ultralytics pre/post), mAP 0.6398 → 0.6349
(Δ −0.0049, parity). First load + compile ≈ 1–2 s (one-time).

**Live demo (NB02 §6)**: the five OpenVINO runtime steps — `ov.Core()` →
`read_model` → `compile_model(CPU, hint)` → `create_infer_request` → `infer`.
Emphasize: **not a magic line of code — a different execution engine.**

**Threads & the bimodal story (advanced, 3 min)**: the container has a
16-core CPU quota; OpenVINO's default spawns one thread per visible core
(64) → cgroup throttling → deterministic 26/75 ms alternation in v2.0.0
data. v2.0.1 pins CPU runs to 16 CPUs (see
`validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.md`). Great story: "the histogram
had two peaks; the kernel's throttle counters explained them."

**Never say**: "OpenVINO is officially certified on AMD CPUs" (official
validation is limited — our numbers are machine-measured) · "AMD has AMX."

---

## Slide 39–42 · INT8 quantization — Notebook 03 (Lab C)

**Core knowledge**: FP32→INT8 = 4× smaller weights; quantization maps ranges
via scale/zero-point; calibration on real data chooses the ranges;
dequantization error is unavoidable.

**Hand example (NB03 §0, run it)**: [0.00, −0.60, 1.20], scale = 1.20/127 →
int8 [0, −64, 127] → dequant [0, −0.605, 1.20]: tiny errors, 4× smaller.

**Measured**: FP32 98.3 ms / 100 MB / mAP 0.6349 → INT8 78.1 ms / 26 MB /
mAP 0.5682 = **1.26× faster, 3.8× smaller, ΔmAP −0.0667**. Quantization
REBUILT this campaign: 150 real train2017 images, NNCF MIXED, 44 s,
identity-verified cache.

**The deployment question (mandatory discussion)**: is 1.26× + 3.8× smaller
worth −0.067 mAP? Answer depends on the business (missed-detection cost vs
server cost). There is no universal answer — that IS the lesson.

**Calibration experiment**: `STUDENT_CALIB_N = 30` re-quantizes into a
separate identity-verified cache (coexists with the 150-image model).

**Never say**: "quantization is lossless" · "INT8 is always faster" (no AMX
on AMD; 1.26× here, could differ elsewhere).

---

## Slide 43–46 · Resolution trade-off — Notebook 04 (Lab D-A)

**Core knowledge**: pixels ∝ imgsz²; 640→512 = 36% fewer pixels → less
compute, but small-object detail is lost.

**Measured (W7900, e2e batch 1)**: 13.68 ms → 13.50 ms = **only −1.3%** e2e,
mAP 0.6398 → 0.6159 (**−0.024**). Forward-only compute scales more; fixed
pre/post + launch overhead dominates e2e at batch 1 — a *measured nuance*,
not a failure of the theory.

**Interaction**: students predict the latency drop (typical: −30%); then
explain the gap with the Amdahl/fixed-overhead story from Slide 30.

---

## Slide 47–48 · Batch & throughput semantics — Notebook 04 (Lab D-B)

**Core knowledge**: batch amortizes per-launch overhead and raises
utilization; VRAM grows ~linearly with batch × imgsz²; **latency semantics
change**.

**Measured (forward-only, W7900)**: b1 10.5 ms → 95.2 img/s; b8 55.2
ms/batch → 144.9 img/s = **1.52×**. End-to-end (labeled separately): b1
13.6 ms per request; b8 80.0 ms per 8-image request.

**The sentence to write on the board**: 55.2 ÷ 8 = 6.9 ms is an *equivalent
per-image compute cost* — **no user ever waits 6.9 ms**; a user in a full
batch waits up to the whole batch time.

**Interaction**: "Your boss saw '1.52× throughput' and promises customers
1.52× lower latency. What do you tell them?"

---

## Slide 49–50 · CPU+GPU pipeline — Notebook 05 (Lab E)

**Core knowledge**: overlap CPU work (load + letterbox) with GPU compute via
a bounded queue (double buffering); throughput → max(stage) × N direction;
per-request latency under burst INCLUDES queue wait.

**Measured (100-image burst)**: serial 58.6 img/s → pipeline 83.1 img/s =
**1.42×**; per-detection parity **100/100 images, min IoU 1.0, max Δconf
0.0** (bit-identical). Serial per-image e2e P50 = 14.0 ms (no queue);
pipeline burst e2e P50 = 611 ms (queue-inclusive) — **different scopes,
never subtracted**.

**The real timeline (NB05 §5b)**: stage timeline drawn from measured
per-image records, not an idealized diagram. Students identify: which stages
interleave (CPU prepare of i+1 under GPU infer of i), the bottleneck (GPU
infer ~10 ms ≫ CPU prepare ~3 ms), and why gain is bounded (~max/sum ratio).

**Two scenarios (must be taught separately)**:
1. *Burst of 100 images* → compare total throughput (this lab).
2. *Steady arrival of single requests* → users care about their own wait;
queueing can make it WORSE than serial.

**Interaction**: "If CPU prepare took 15 ms and GPU infer 5 ms, would the
pipeline still help?" (Bottleneck flips to CPU; gain bounded by CPU stage.)

**Never say**: "async queue ⇒ overlap ⇒ speedup" (overlap must be shown with
stage timelines); "detections identical" without the per-detection check.

---

## Slide 51 · Six conclusions — all notebooks

| # | Claim | Number (this round) | Cost/uncertainty |
|---|---|---|---|
| 1 | W7900 vs EPYC 9334 (e2e, same model) | **14.14×** | platform comparison; spread ≤ 7.5% |
| 2 | OpenVINO vs PyTorch (same CPU) | **2.28×** | mAP parity (−0.0049); one-time compile |
| 3 | INT8 vs FP32 (same CPU) | **1.26×**, 3.8× smaller | **ΔmAP −0.0667** — NOT lossless |
| 4 | imgsz 640→512 | −1.3% e2e | mAP −0.024; fixed overhead dominates |
| 5 | batch 1→8 (forward) | **1.52×** throughput | e2e 13.6 → 80.0 ms/8-img request; ÷8 ≠ latency |
| 6 | async CPU+GPU pipeline | **1.42×** burst throughput | detections bit-identical; burst queue raises per-request wait |

**How proven**: campaign id + run ids on every chart; CSVs immutable;
SHA256SUMS verified; independent audit recomputes every number
(`AUDIT_REPORT.md`); reproduce with `scripts/run_campaign.py`.

---

## Statements to avoid (teaching integrity) / 不能说的话

- ❌ model latency = end-to-end latency (they differ by fixed pre/post).
- ❌ batch time ÷ batch size = user latency.
- ❌ CPU-vs-GPU difference = software speedup (it is a platform comparison).
- ❌ INT8 is accuracy-lossless (ΔmAP −0.0667 measured here).
- ❌ the 120-image educational subset is "COCO mAP" (it is not the official
  benchmark; say "educational subset" every time).
- ❌ async queue = proven overlap (only stage timelines prove overlap).
- ❌ AMD EPYC has Intel AMX (it has AVX-512/VNNI; no AMX).

## Student-experience notes (R3.9 review)

Every student knob was re-executed this round and verifiably changes the
output: `STUDENT_IMAGE_INDEX` (new image detection), `STUDENT_DEVICE`
(CPU/GPU P50), `STUDENT_HINT` (real ov.Core compile with different hint),
`STUDENT_CALIB_N` (separate identity-verified INT8 cache), `STUDENT_IMGSZ_B`
(new latency/mAP/VRAM row), `STUDENT_QUEUE_DEPTH`/`STUDENT_N_IMAGES`
(pipeline re-run). No knob reads pre-generated charts.
