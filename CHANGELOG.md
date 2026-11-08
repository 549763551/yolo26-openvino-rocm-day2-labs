# Changelog

## Unreleased — v2.0.2 release-evidence accuracy corrections (documentation/QA only)

- Correct published P95 values by recomputing per-round percentiles from the sealed v2.0.1 campaign; original CSVs and campaign hashes remain untouched. In particular Lab A CPU P95 = **470.6 ms**, not 203.1 ms; Lab B PyTorch CPU P95 = **441.0 ms**, not 199.4 ms.
- Resolve conflict markers and reconcile the exact Slide 28–51 map with the approved teaching outline.
- Explicitly label FP32/INT8 sizes as historical v2.0.0 exports; v2.0.1 had no freshly captured model-size artifacts. Add a local XML/BIN measurement tool (no GPU needed) for the missing fresh evidence.
- Correct missing cold-start log reference; disclose that only the audit summary, not the raw installer transcript, is public.
- Add release-documentation tests driven from original campaign CSVs and wire them into CI. No new W7900 physical benchmark is claimed.



## v2.0.1 — 2026-11-08 — Quality re-audit, benchmark revalidation, teaching upgrade

Independent audit pass (R1–R4) over v2.0.0. All six experiments **re-measured
on real hardware** into campaign `validation/v2.0.1/campaigns/c20261108_1108`
(3 rounds × 50 samples, median of rounds, independently audited). Historical
v2.0.0 data preserved under `results/raw/` (reference only).

### Fixed (P0)
- **Data-skip defect (B-01)**: `setup.sh` treated the committed
  `data/manifest.json` as proof of downloaded images — a fresh clone skipped
  preparation. Setup now runs `scripts/verify_data.py` (real image counts,
  labels, yaml, calib list, per-file SHA256, train/val isolation) and repairs
  incrementally; negative tests cover manifest-only clones.
- **Self-hash defect (B-02)**: `save_run()` hashed the summary JSON, then
  rewrote the file with the hash inside — recorded hash never matched bytes.
  New evidence chain: immutable CSV → summary referencing CSV SHA256 →
  directory SHA256SUMS (written after final bytes) →
  `scripts/verify_evidence.py --campaign ...` recomputes everything; tamper
  tests included.
- **Lab E count-only comparison (B-03)**: `compare_results()` compared
  `len(boxes)` only. Now per-image, per-detection one-to-one matching
  (class + box IoU ≥ 0.95 + conf ± 0.02, greedy), with exact extras/misses
  reported; this round measured 100/100 images bit-identical (min IoU 1.0,
  max Δconf 0.0).
- **No notebook execution gate (B-04)**: added
  `scripts/validate_notebooks.sh --execute` (nbclient runs every cell in the
  project kernel, verifies expected outputs, static model-rule gate against
  smaller/fake models, writes `EXECUTION_MATRIX.md`). 6/6 notebooks · 45/45
  code cells this release.

### Fixed (P1)
- **INT8 cache identity (B-05)**: cache hit was "xml exists". Cache dirs are
  now keyed by the full identity (model sha, FP32 export sha, calibration
  manifest sha + count, NNCF preset/version, subset size, imgsz); 150- and
  30-image models coexist; stale reuse impossible.
- **CPU topology (B-06)**: `threads` was threads_per_core × sockets (=4).
  Now full topology: sockets/cores/threads, total physical + logical CPUs,
  cgroup quota (16 cores) vs physical (64) reported separately, affinity,
  NUMA layout; explicit "AMD has no Intel AMX" note.
- **latest.json contamination (B-07)**: partial reruns silently merged old
  results. Fresh by default; `--resume` is explicit. Formal numbers now come
  from campaign-scoped CAMPAIGN.json (single campaign id per set of claims).
- **CPU bimodal latency diagnosed (B-08)**: root cause = cgroup CPU
  throttling (OV default 64 threads vs 16-core quota), proven via
  `cpu.stat` throttle counters; fixed in benchmarks by pinning CPU runs to 16
  quiet CPUs. Reproducible: `scripts/diagnose_cpu_bimodal.py` →
  `validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.{md,json}`.

### Added
- `scripts/run_campaign.py` — formal multi-round benchmark campaigns with
  per-round host-load/throttle recording, fresh INT8 rebuild, per-detection
  Lab E evidence.
- `scripts/audit_benchmarks.py` — independent recomputation of every public
  number from raw CSV bytes (53 runs, 96 files hashed: PASS).
- `scripts/make_campaign_charts.py` — the six slide charts regenerated from
  the campaign with evidence footnotes (campaign/run ids, scope, SHA).
- Teaching upgrades (R3): Amdahl mini-lab (NB01), real `ov.Core()` runtime
  demo + fake STUDENT_HINT knob fixed (NB02), five-number INT8 example +
  deployment question (NB03), resolution/batch timeline semantics (NB04),
  measured stage-timeline chart + two-scenario latency framing (NB05),
  dual-path architecture diagram (NB00); bilingual Predict / Troubleshooting
  sections everywhere; instructor guide rewritten for slides 28–51.
- All student knobs verified to actually change measured outputs
  (`validation/v2.0.1/STUDENT_KNOB_VERIFICATION.md`).

### Measured changes vs v2.0.0 (same machine, new protocol — honest deltas)

| Experiment | v2.0.0 | v2.0.1 (campaign c20261108_1108) |
|---|---|---|
| CPU vs GPU | 9.40× | **14.14×** (191.3→13.5 ms P50) |
| PyTorch→OpenVINO CPU | 3.09× | **2.28×** (198.5→87.2 ms) |
| FP32→INT8 | 1.75× | **1.26×** (98.3→78.1 ms) · ΔmAP −0.0667 (unchanged, deterministic) |
| resolution 640→512 | −5% e2e | **−1.3%** e2e |
| batch 1→8 forward | 1.49× | **1.52×** (95.2→144.9 img/s) |
| serial→async pipeline | 1.21× | **1.42×** (58.6→83.1 img/s) |

Deltas stem from shared-host load (now recorded per round), quota-matched CPU
pinning (throttling removed), and the multi-round median protocol — all
disclosed in `results/RESULTS_TABLE.md`.

## v2.0.0 — 2026-11-08 — Initial public release

Six bilingual notebooks, six real experiments on 2× EPYC 9334 + W7900,
one-click setup, slide charts, evidence CSVs. Known defects B-01..B-10
documented in `validation/v2.0.1/00_BASELINE_AUDIT.md` (fixed in v2.0.1).
