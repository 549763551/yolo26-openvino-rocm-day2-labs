# Student Experience Review (R3.9) — v2.0.1

**Method disclosure**: this is an *engineer-simulated student review* — the
reviewer executed all six notebooks end-to-end from the perspective of a
senior undergraduate who knows Python basics but no ROCm/OpenVINO/quantization
background. It is NOT a real user study with external students; that remains
future work. Evidence for every score is cited from this round's runs.

Scoring: 1 (poor) – 5 (excellent) per dimension.

## Per-notebook review

### 00 — AMD Environment & Official YOLO26l
- Readability 5 — bilingual 10-section template; architecture diagram shows
  the two runtime paths (CPU plugin vs ROCm GPU) explicitly.
- Understandability 5 — "which device is it really running on" is answered
  with code + diagram, not assertions.
- Operability 5 — executed 7/7 cells, 7 s (EXECUTION_MATRIX.md).
- Real feedback 5 — prints real vendor/model/sha256/GPU name; a real detection
  with boxes drawn.
- Debugging 4 — troubleshooting section covers the 3 classic misconceptions
  (OpenVINO≠GPU, torch.cuda≠NVIDIA, identity before performance).
- Theory↔result 4 — identity/hardware concepts map 1:1 to outputs.
- Career value 4 — SHA-verified model identity + topology awareness are real
  Mlops habits.

### 01 — Lab A CPU vs GPU
- Readability 5; Understandability 5 — P50/P95/throughput/warm-up defined
  before measurement.
- Operability 5 — 9/9 cells, 13 s.
- Real feedback 5 — live CPU vs GPU table + chart from the run itself.
- **New interactive Amdahl mini-lab (v2.0.1)**: students hand-compute
  30 ms → 25 ms before running; the measured model-only vs e2e gap
  (~10.5 vs ~13.5 ms) grounds it in real data. Predict→run→explain loop 5.
- Debugging 5 — sync/warm-up/P95 mistakes listed.
- Theory↔result 5 — connects directly to Slide 30.

### 02 — Lab B OpenVINO on AMD CPU
- Readability 5; Understandability 5.
- Operability 5 — 7/7 cells, 39–56 s.
- Real feedback 5 — same-CPU A/B with mAP parity shown next to latency.
- **v2.0.1 fix**: the STUDENT_HINT knob previously did nothing (property set
  on an unused Core). Now a real 5-step `ov.Core()` demo compiles with the
  chosen hint and measures model-only latency — the knob verifiably changes
  the result (see STUDENT_KNOB_VERIFICATION.md). 5.
- Debugging 5 — includes the AMX misconception fix (AMD Zen4: AVX-512/VNNI,
  no AMX) and the bimodal-latency/throttling story.
- Theory↔result 5 — "runtime ≠ magic code swap" shown with real IR loading.

### 03 — Lab C INT8
- Readability 5; Understandability 5.
- **New five-number quantization example (v2.0.1)**: FP32 [0, −0.60, 1.20] →
  scale → int8 → dequant → error, run in one cell; makes ΔmAP intuitive. 5.
- Operability 5 — 9/9 cells, 30 s (first-ever quantization +44 s, disclosed).
- Real feedback 5 — size/latency/mAP three-panel comparison.
- **v2.0.1 fix**: STUDENT_CALIB_N 150→30 now builds a *separate
  identity-verified* cache (old behavior silently reused the 150-image
  model). Verified this round. 5.
- Debugging 5 — "never say lossless", calibration/val isolation, INT8 not
  guaranteed faster.
- The mandatory deployment question ("is 1.26× + 3.8× smaller worth
  −0.067 mAP?") makes the trade-off explicit. 5.

### 04 — Lab D resolution & batch
- Readability 5; Understandability 5 — causal chain for resolution + batch
  timeline ASCII diagram (v2.0.1).
- Operability 5 — 6/6 cells, 25 s.
- Real feedback 5 — latency+mAP+VRAM table; batch throughput with explicit
  "55.2÷8 = equivalent cost, NOT request latency" labeling.
- STUDENT_IMGSZ_B 416/512/768 re-measures live (verified). 5.
- Debugging 4 — the three classic batch/resolution mistakes covered.

### 05 — Lab E CPU+GPU pipeline
- Readability 5; Understandability 5 — two-scenario latency framing
  (burst throughput vs per-request wait) is now explicit (v2.0.1).
- Operability 5 — 7/7 cells, 10 s.
- Real feedback 5 — **real stage timeline from measured per-image records**
  (not an idealized diagram): students SEE CPU-prepare overlapping GPU-infer;
  bottleneck question answered from pre_p50 vs infer_p50.
- Correctness 5 — per-detection parity (100/100 images, min IoU 1.0,
  max Δconf 0.0) with the rule and justification printed.
- STUDENT_QUEUE_DEPTH 1/8 re-runs the pipeline (verified). 5.
- Debugging 5 — "queue ≠ overlap proof", "never subtract the two e2e scopes".

## Summary scores (1–5)

| Dimension | 00 | 01 | 02 | 03 | 04 | 05 | avg |
|---|---|---|---|---|---|---|---|
| Readability | 5 | 5 | 5 | 5 | 5 | 5 | 5.0 |
| Understandability | 5 | 5 | 5 | 5 | 5 | 5 | 5.0 |
| Operability (one-click run) | 5 | 5 | 5 | 5 | 5 | 5 | 5.0 |
| Real-result feedback | 5 | 5 | 5 | 5 | 5 | 5 | 5.0 |
| Debugging/troubleshooting | 4 | 5 | 5 | 5 | 4 | 5 | 4.7 |
| Theory↔result connection | 4 | 5 | 5 | 5 | 4 | 5 | 4.7 |
| Career/project value | 4 | 4 | 5 | 5 | 4 | 5 | 4.6 |

Weakest points (honest): NB04 lacks a per-notebook troubleshooting code-level
debug exercise (mitigated by docs/TROUBLESHOOTING.md); career value is
inherent to the topic, not the notebook design.

All student knobs verified to actually change outputs this round:
`STUDENT_KNOB_VERIFICATION.md` (5/5 knobs verified).
