# Benchmark Protocol / 性能测量协议

This document defines exactly how every number in this repository is produced.
All six labs share the harness in `course/benchmark.py`. Nothing is measured
differently per-notebook.

## Model identity / 模型身份

- Single source: **official Ultralytics `yolo26l.pt`** (COCO detection, 80
  classes), downloaded from the Ultralytics release asset URL, SHA256 recorded
  in `weights/model_identity.json`.
- OpenVINO FP32 and INT8 are exported from this **same file**. No other
  weights are ever used for benchmarking (no n/s/m/x, no fine-tuned models).
- Verification at import time: `course.model.load_official_yolo26l()` asserts
  80 classes + `detect` task.

## Machine / 机器

Recorded automatically in every `*_summary.json` (`meta.cpu`, `meta.gpu`,
`meta.env`): AMD CPU model, GPU name, torch/HIP/OpenVINO/Ultralytics versions.
Reference machine: 2× AMD EPYC 9334, Radeon PRO W7900 (gfx1100).

## Data / 数据

- Latency benchmarks: first image of the fixed `data/val_subset` (120 COCO
  val2017 images, seed=42 selection, manifest with SHA256).
- mAP: full `data/val_subset` via ultralytics `.val()` (batch 8).
- INT8 calibration: `data/calib/images` (150 COCO train2017 images,
  **disjoint** from val by construction).
- Label: *Educational COCO Subset — Not Official COCO Benchmark*.

## Timing rules / 计时规则

1. **Warm-up excluded**: `quick` = 3 warm-up + 10 timed iterations;
   `full` = 10 warm-up + 50 timed. Counts recorded in every summary.
2. **GPU asynchrony**: every timed GPU call is bracketed by
   `torch.cuda.synchronize()` (ROCm/HIP). CPU calls are synchronous already.
3. **First-load separation**: model loading, OpenVINO export and NNCF
   quantization are timed and reported as one-time costs, never mixed into
   per-inference latency.
4. **Latency scope**: "end-to-end" = ultralytics `predict()` on one image
   (letterbox + normalize + forward + postprocess). "GPU forward" = model
   forward on a pre-built tensor (Lab D-B batch experiments); scope is stated
   in each chart/summary `notes` field.
5. **Statistics**: mean, P50 (median), P95, min, max over all timed samples.
   Raw samples are written to CSV; **no samples are ever dropped**, including
   slow ones.
6. **Throughput** = images ÷ wall time (batch: batch_size ÷ batch time).
   Batch-time ÷ batch_size is reported only as *equivalent per-image compute
   cost*, explicitly never as per-request latency.
7. **Repetition across runs**: the `full` profile can be executed multiple
   times; each run is a separate timestamped CSV/JSON set.

## Memory / 内存

- GPU: `torch.cuda.max_memory_allocated()` peak between reset and read
  (torch allocator view, in GB).
- Host RAM: peak RSS of the process (`ru_maxrss`), reported where relevant
  (Lab C records process-level peak RSS during FP32 vs INT8 inference loops).

## Accuracy / 精度

- mAP50-95 (and mAP50/mAP75) via `ultralytics .val()` on the fixed educational
  subset, FP32, batch 8, imgsz as specified. INT8 mAP measured with the
  quantized OpenVINO model on CPU.
- ΔmAP is always reported next to any speedup (see Lab C chart).

## Honesty rules / 诚实规则

- Results may be "no speedup" or "slower" — they are still results.
- No swapping models, images, or thread configs between the two sides of a
  comparison. OpenVINO CPU thread/config explorations are clearly labeled as
  extra experiments, separate from the default fair comparison.
- OpenVINO's official validation on AMD CPUs is limited; everything here is
  *measured on this machine* and claimed only for this machine class.

## Evidence format / 证据格式

```
results/raw/<experiment>/<run_id>.csv            # iteration,ms
results/raw/<experiment>/<run_id>_summary.json   # meta + stats + evidence sha256
results/latest.json                              # merged view for charts/README
```

Every chart in `assets/slides/` embeds the run id, the CSV path and its
SHA256 prefix, so any figure can be regenerated and audited from raw data.

---

## v2.0.1 additions — formal campaigns & evidence chain

**Formal campaigns** (`scripts/run_campaign.py`, e.g.
`validation/v2.0.1/campaigns/c20261108_1108/`):

- **Rounds**: every latency variant is measured in **3 independent rounds ×
  50 timed samples** (warm-up 10 excluded in each round); published P50/P95
  are the **median across rounds**, with per-round values and spread recorded
  in `CAMPAIGN.json → aggregate`.
- **Interleaving**: PyTorch/OpenVINO order alternates between rounds
  (A/B then B/A) to cancel time drift.
- **CPU protocol**: process pinned to 16 quiet CPUs (one NUMA node, re-picked
  per round, recorded per run in `meta.extra.affinity_cpus`) matching the
  16-core **cgroup quota**; PyTorch CPU threads pinned to 8 (measured optimum
  under the quota). Rationale: >16 threads triggers cgroup throttling and a
  deterministic bimodal latency — see
  `validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.md` (throttling proven via
  `/sys/fs/cgroup/cpu.stat` counters).
- **Shared-host disclosure**: host load average and cgroup throttle counters
  are recorded around every round (`CAMPAIGN.json → round_env`). Numbers are
  medians of rounds; per-round spread is published. We never claim exclusive
  hardware.
- **Fresh quantization**: Lab C rebuilds the INT8 model in-campaign with the
  full calibration manifest recorded in `weights/int8_cache/*/CACHE_KEY.json`
  (identity key: model sha + FP32 export sha + calibration dataset sha +
  count + preset + NNCF version + subset size + imgsz).
- **Lab E evidence**: per-image stage timings AND per-detection outputs are
  stored as records JSON; the throughput gain is only published when
  per-detection parity (class + IoU ≥ 0.95 + conf ± 0.02) passes for every
  image; serial e2e (no queue) and pipeline burst e2e (queue-inclusive) are
  reported as **separate scopes**, never subtracted.

**Evidence chain** (fixes the v2.0.0 self-hash defect):

1. raw per-iteration CSV — immutable measurement record;
2. `*_summary.json` — references the CSV sha256, never its own;
3. directory `SHA256SUMS` — written after final bytes, covers CSVs,
   summaries and records; campaign root `SHA256SUMS` covers everything;
4. `scripts/verify_evidence.py --campaign <dir>` — recomputes every hash and
   statistic from bytes (tamper-proof verification);
5. `scripts/audit_benchmarks.py --campaign <dir>` — independent recomputation
   of P50/P95/throughput/speedups + config invariants (model sha, imgsz,
   batch, device, warm-up exclusion) + parity + mAP cross-checks; writes
   `AUDIT_REPORT.md`.

**Batch semantics**: forward-only throughput is always labeled separately
from end-to-end; batch time ÷ batch size is an *equivalent per-image compute
cost*, never a request latency (both are measured).
