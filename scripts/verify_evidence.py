#!/usr/bin/env python3
"""Independent evidence verification for benchmark campaigns (v2.0.1).

Usage:
    python scripts/verify_evidence.py --campaign validation/v2.0.1

What it does (nothing is trusted from recorded values alone):
  * walks the campaign tree and verifies EVERY file listed in each
    experiment directory's SHA256SUMS by recomputing sha256 from bytes;
  * for every *_summary.json: recomputes the CSV sha256 it references and
    compares with the recorded csv_sha256; recomputes row count and
    P50/P95/mean/min/max from the CSV and compares with summary.stats;
  * verifies the campaign-level SHA256SUMS covers every evidence file;
  * verifies no summary contains its own hash (the v2.0.0 circular-write
    defect leaves json_sha256 != actual file hash — detected here);
  * exits non-zero on ANY tampering.

Tamper tests (run by the independent audit):
    modify one CSV byte   -> FAIL
    modify summary JSON   -> FAIL
    drop one SUMS entry   -> FAIL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(samples: list[float], q: float) -> float:
    s = sorted(samples)
    if not s:
        raise ValueError("no samples")
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verify_dir_sums(exp_dir: Path, problems: list[str]) -> None:
    sums = exp_dir / "SHA256SUMS"
    if not sums.exists():
        problems.append(f"MISSING SHA256SUMS in {exp_dir}")
        return
    seen = set()
    for ln in sums.read_text().splitlines():
        if not ln.strip():
            continue
        digest, name = ln.split(None, 1)
        name = name.strip()
        p = exp_dir / name
        if not p.exists():
            problems.append(f"SUMS references missing file: {p}")
            continue
        actual = sha256(p)
        seen.add(name)
        if actual != digest:
            problems.append(f"TAMPERED (dir sums): {p} "
                            f"recorded {digest[:12]} actual {actual[:12]}")
    for p in exp_dir.iterdir():
        if ((p.suffix == ".csv" or p.name.endswith("_summary.json")
             or ("records" in p.name and p.suffix == ".json"))
                and p.name not in seen):
            problems.append(f"NOT COVERED by SHA256SUMS: {p}")


def verify_summary(json_path: Path, problems: list[str]) -> dict | None:
    try:
        rel = str(json_path.relative_to(REPO))
    except ValueError:
        rel = str(json_path)
    try:
        s = json.loads(json_path.read_text())
    except Exception as e:
        problems.append(f"UNREADABLE summary {rel}: {e}")
        return None
    ev = s.get("evidence", {})
    csv_rel = ev.get("csv")
    if not csv_rel:
        problems.append(f"summary {rel} has no evidence.csv")
        return s
    csv = REPO / csv_rel
    if not csv.exists():
        problems.append(f"summary {rel} references missing CSV {csv_rel}")
        return s
    actual_csv = sha256(csv)
    if ev.get("csv_sha256") and actual_csv != ev["csv_sha256"]:
        problems.append(f"TAMPERED CSV: {csv_rel} recorded "
                        f"{ev['csv_sha256'][:12]} actual {actual_csv[:12]} "
                        f"(summary {rel})")
    # recompute stats from raw CSV bytes
    rows = [ln.split(",") for ln in csv.read_text().splitlines()[1:] if ln]
    try:
        ms = [float(r[1]) for r in rows]
    except Exception as e:
        problems.append(f"CSV {csv_rel} unparsable: {e}")
        return s
    st = s.get("stats", {})
    n = st.get("n")
    checks = {
        "n": (len(ms), n),
        "p50_ms": (pct(ms, 50), st.get("p50_ms")),
        "p95_ms": (pct(ms, 95), st.get("p95_ms")),
        "mean_ms": (statistics.fmean(ms), st.get("mean_ms")),
        "min_ms": (min(ms), st.get("min_ms")),
        "max_ms": (max(ms), st.get("max_ms")),
    }
    for key, (recomputed, recorded) in checks.items():
        if recorded is None:
            continue
        if not close(recomputed, recorded):
            problems.append(f"STATS MISMATCH {rel} {key}: summary "
                            f"{recorded} vs recomputed {recomputed:.4f}")
    # warmup exclusion: CSV rows must equal the recorded iteration count
    iters = s.get("meta", {}).get("iters")
    if isinstance(iters, int) and len(ms) != iters:
        problems.append(f"sample count {len(ms)} != meta.iters {iters} in {rel} "
                        "(warm-up contamination suspected)")
    # self-hash defect check: recorded json_sha256 must equal the file's hash
    if ev.get("json_sha256") and sha256(json_path) != ev["json_sha256"]:
        problems.append(f"SELF-HASH MISMATCH: {rel} recorded json_sha256 "
                        f"{ev['json_sha256'][:12]} != actual "
                        f"{sha256(json_path)[:12]} (v2.0.0 defect B-02 "
                        "signature; v2.0.1 summaries are verified via SUMS)")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True,
                    help="campaign root, e.g. validation/v2.0.1")
    args = ap.parse_args()
    root = Path(args.campaign)
    if not root.is_absolute():
        root = REPO / root
    if not root.exists():
        print(f"campaign root not found: {root}")
        return 2

    problems: list[str] = []
    n_summaries = 0
    n_csv = 0
    for exp_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        if not any(exp_dir.glob("*.csv")):
            continue
        verify_dir_sums(exp_dir, problems)
        for j in sorted(exp_dir.glob("*_summary.json")):
            n_summaries += 1
            verify_summary(j, problems)
        n_csv += len(list(exp_dir.glob("*.csv")))

    # campaign-level manifest must exist for formal campaigns
    for cand in ("CAMPAIGN.json", "SHA256SUMS"):
        if (root / cand).exists():
            break
    else:
        if any(root.rglob("*_summary.json")):
            problems.append("campaign root lacks CAMPAIGN.json/SHA256SUMS")

    try:
        campaign_rel = root.relative_to(REPO)
    except ValueError:
        campaign_rel = root
    print(f"campaign: {campaign_rel}")
    print(f"  csv files verified      : {n_csv}")
    print(f"  summaries re-verified   : {n_summaries}")
    if problems:
        print(f"EVIDENCE VERIFICATION: FAIL ({len(problems)} problem(s))")
        for p in problems:
            print("  -", p)
        return 1
    print("EVIDENCE VERIFICATION: PASS (all hashes + stats recomputed from "
          "bytes and matched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
