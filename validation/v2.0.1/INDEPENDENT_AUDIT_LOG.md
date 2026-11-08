# R4 — Independent Release Audit Log (v2.0.1)

Independent verification performed in a **fresh clone** at
`/workspace/audit_v201/repo` (NOT the development tree), on frozen candidate
lineage `ca344b3` (+ final audit-fix commit `e5af233`: verify_evidence
container-directory handling — audit tooling only, no production-code change;
CI re-ran green on it before merge).

Environment independence:
- fresh `git clone` from GitHub at the frozen SHA;
- venv + weights + data provisioned via verified transport cache (pip wheel
  cache / sha256-checked copies of the official yolo26l.pt and the
  manifest-verified COCO subsets — all re-verified from bytes by the battery
  below);
- OpenVINO FP32 export and INT8 quantization were deliberately NOT copied —
  the audit tree re-exported and re-quantized from the official weights,
  proving the pipeline reproduces from weights alone.

## Audit battery results (fresh clone)

| # | Check | Result |
|---|---|---|
| 1 | `scripts/verify_data.py` | PASS — 120 val + 150 calib, sha256 match, train/val disjoint |
| 2 | `scripts/doctor.sh` (incl. weight sha recomputed from bytes) | ALL HARD GATES PASS; torch/torchvision pins verified 2.12.0/0.27.0+rocm7.14.1 |
| 3 | `scripts/verify_evidence.py --campaign validation/v2.0.1` | PASS — every SHA256SUMS + summary stat recomputed from bytes |
| 4 | `scripts/audit_benchmarks.py` (independent recompute) | PASS — 53 runs, 96 files, all headline ratios/mAP/parity verified |
| 5 | `pytest tests/test_units.py tests/test_v201_fixes.py` | 19/19 passed |
| 6 | model identity vs campaign | MATCH — weights sha256 == CAMPAIGN.json sha == `9fe3c544…db26` |
| 7 | README headline numbers vs campaign JSON | MATCH — 14.14 / 2.28 / 1.26 / 1.52 / 1.42 / 0.6398 / 0.5682 all present and equal |
| 8 | Notebook EXECUTION gate (real nbclient runs, local kernel) | **6/6 notebooks · 45/45 code cells PASS** (fresh export + fresh quantization inside NB02/NB03) |
| 9 | Fault injection battery | **9/9 detected** (table below) |
| 10 | Open-source hygiene | 0 secrets in full git history (token/AWS/private-key patterns); no weights/data/venv committed; LICENSE (MIT) present; README links valid |

Note on audit-tree notebook runs: two earlier executions showed a ZMQ
"Socket operation on non-socket" failure on NB02 when the executor picked a
kernelspec belonging to a *different* tree's venv (user-level registration).
Registering a kernelspec for the audit tree's own interpreter resolved it —
6/6 PASS. Root cause was audit-harness setup, not notebook code; dev and
cold-start trees were unaffected.

## Fault injections (all must be DETECTED)

| # | Injection | Expected | Observed |
|---|---|---|---|
| F1 | modify one CSV byte | evidence FAIL | DETECTED (dir SHA256SUMS + summary csv hash) |
| F2 | tamper summary stats | evidence FAIL | DETECTED (stats recomputed from CSV mismatch) |
| F3 | delete one val image | data integrity FAIL | DETECTED (missing + count mismatch) |
| F4 | alter recorded weight sha | identity FAIL | DETECTED (doctor recomputes sha from bytes) |
| F5 | swap one detection box/class | parity FAIL | DETECTED (per-detection matcher reports extras/misses) |
| F6 | INT8 calibration 150→30 | distinct cache identity | VERIFIED (separate identity-keyed dirs) |
| F6b | stale CACHE_KEY.json | must rebuild | VERIFIED (wrong key rejected, model rebuilt) |
| F7 | stale run id mixed into campaign | campaign audit FAIL | DETECTED (run-id membership check vs CAMPAIGN.json) |
| F8 | unknown notebook kernel | execution gate FAIL | DETECTED (kernel resolution rejects) |

## Four release gates

| Gate | Verdict | Basis |
|---|---|---|
| Engineering | **GO** | fresh-clone setup 10/10 + doctor PASS + 19/19 tests + 6/6 notebooks 45/45 cells (three independent trees: dev, coldstart, audit) + CI green on release branch |
| Benchmark | **GO** | all six experiments re-measured on real EPYC 9334 + W7900D (campaign c20261108_1108, 3×50 samples); independent recomputation PASS; per-detection parity 100/100 (IoU 1.0); INT8 identity verified + rebuilt; bimodal latency root-caused (cgroup throttling) and disclosed; shared-host load recorded |
| Teaching | **GO** | bilingual 10-section template complete; 6/6 executed; 5/5 student knobs verified to change outputs; instructor guide covers slides 28–51 with forbidden-claims list; engineer-simulated review disclosed as such |
| Open Source | **GO** | MIT license intact; no secrets in history; no large artifacts committed; CI green (incl. offline chart re-render job); all referenced paths exist; evidence sealed with SHA256SUMS + independent audit report |

All four gates GO → v2.0.1 released.
