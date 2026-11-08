#!/usr/bin/env python3
"""Independent benchmark audit (R2.6): recompute EVERYTHING from raw bytes.

Deliberately does NOT import course/ measurement code paths that produced the
numbers (only re-implements percentile math). For a campaign directory it:

  1. verifies every SHA256SUMS entry (campaign root + per-experiment dirs)
     by recomputing hashes from file bytes;
  2. recomputes per-run n/mean/P50/P95/min/max from each CSV and compares
     with the *_summary.json stats;
  3. verifies each summary's csv_sha256 against the CSV bytes;
  4. verifies sample counts equal meta.iters (warm-up exclusion);
  5. verifies per-experiment invariants: identical model_sha256 across
     variants, expected imgsz/batch/device/runtime per experiment;
  6. recomputes every headline speedup/ratio in CAMPAIGN.json from the
     median-of-rounds P50s and compares with the published values;
  7. verifies Lab E parity files exist and are marked consistent;
  8. verifies mAP records exist for every accuracy-claiming lab;
  9. writes AUDIT_REPORT.md and exits non-zero on any discrepancy.

Usage:
    python scripts/audit_benchmarks.py --campaign validation/v2.0.1/campaigns/<id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- independent percentile implementation (linear interpolation) ---------
def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def sha256b(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def close(a, b, tol=5e-3) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


EXPECT = {
    "labA_cpu_vs_gpu": {"pytorch_cpu": {"device": "cpu", "imgsz": 640},
                        "pytorch_rocm_gpu": {"device": "cuda(gfx1100)",
                                             "imgsz": 640}},
    "labB_runtime_cpu": {"pytorch_cpu": {"device": "cpu", "imgsz": 640},
                         "openvino_cpu": {"device": "cpu", "imgsz": 640}},
    "labC_int8": {"openvino_fp32": {"device": "cpu", "imgsz": 640,
                                     "precision": "fp32"},
                  "openvino_int8": {"device": "cpu", "imgsz": 640,
                                    "precision": "int8"}},
    "labD_resolution": {"imgsz640": {"imgsz": 640,
                                     "device": "cuda(gfx1100)"},
                        "imgsz512": {"imgsz": 512,
                                     "device": "cuda(gfx1100)"}},
    "labD_batch": {"batch1_forward": {"batch": 1},
                   "batch8_forward": {"batch": 8},
                   "batch1_e2e": {"batch": 1},
                   "batch8_e2e": {"batch": 8}},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    root = Path(args.campaign)
    if not root.is_absolute():
        root = REPO / root
    if not root.exists():
        print(f"campaign not found: {root}")
        return 2
    problems: list[str] = []
    checks: list[str] = []

    campaign = json.loads((root / "CAMPAIGN.json").read_text())
    model_sha = campaign["model"]["sha256"]

    # 1. root + dir SHA256SUMS
    n_files = 0
    root_sums = root / "SHA256SUMS"
    if root_sums.exists():
        for ln in root_sums.read_text().splitlines():
            if not ln.strip():
                continue
            d, name = ln.split(None, 1)
            p = root / name.strip()
            if not p.exists():
                problems.append(f"root SUMS: missing {name}")
                continue
            n_files += 1
            if sha256b(p) != d:
                problems.append(f"TAMPERED (root sums): {name}")
    else:
        problems.append("campaign root SHA256SUMS missing")

    # 2-5. per-run recomputation + invariants
    runs: dict[str, dict[str, list[float]]] = {}
    run_meta: dict[str, list[dict]] = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()
                          and p.name != "maps"):
        summaries = sorted(exp_dir.glob("*_summary.json"))
        if not summaries:
            continue
        ds = exp_dir / "SHA256SUMS"
        if ds.exists():
            for ln in ds.read_text().splitlines():
                if not ln.strip():
                    continue
                d, name = ln.split(None, 1)
                p = exp_dir / name.strip()
                if not p.exists():
                    problems.append(f"dir SUMS: missing {exp_dir.name}/{name}")
                elif sha256b(p) != d:
                    problems.append(f"TAMPERED (dir sums): "
                                    f"{exp_dir.name}/{name}")
        else:
            problems.append(f"{exp_dir.name}: no SHA256SUMS")
        for sp in summaries:
            s = json.loads(sp.read_text())
            rid = s["run_id"]
            meta = s["meta"]
            variant = meta["variant"]
            # model identity
            if meta.get("model_sha256") and meta["model_sha256"] != model_sha:
                problems.append(f"{rid}: model sha {meta['model_sha256'][:12]} "
                                f"!= campaign {model_sha[:12]}")
            # expected config
            exp_cfg = EXPECT.get(exp_dir.name, {}).get(variant)
            if exp_cfg:
                for k, v in exp_cfg.items():
                    if meta.get(k) != v:
                        problems.append(f"{rid}: meta.{k}={meta.get(k)!r} "
                                        f"expected {v!r}")
            # CSV recompute
            csv = REPO / s["evidence"]["csv"]
            rows = [l.split(",") for l in csv.read_text().splitlines()[1:]
                    if l.strip()]
            ms = [float(r[1]) for r in rows]
            st = s["stats"]
            recomputed = {"n": len(ms), "mean_ms": statistics.fmean(ms),
                          "p50_ms": pctl(ms, 50), "p95_ms": pctl(ms, 95),
                          "min_ms": min(ms), "max_ms": max(ms)}
            for k, v in recomputed.items():
                if not close(v, st.get(k, -1)):
                    problems.append(f"{rid}: stats.{k} summary={st.get(k)} "
                                    f"recomputed={v:.4f}")
            if sha256b(csv) != s["evidence"]["csv_sha256"]:
                problems.append(f"{rid}: csv_sha256 mismatch vs bytes")
            iters = meta.get("iters")
            if isinstance(iters, int) and len(ms) != iters:
                problems.append(f"{rid}: {len(ms)} samples != iters {iters}")
            if meta.get("warmup", 0) < 0:
                problems.append(f"{rid}: negative warmup")
            runs.setdefault(exp_dir.name, {}).setdefault(variant, []).extend(ms)
            run_meta.setdefault(exp_dir.name, {}).setdefault(variant, []
                                                             ).append(s)
            checks.append(f"{rid}: n={len(ms)} p50={recomputed['p50_ms']:.2f}"
                          f" p95={recomputed['p95_ms']:.2f} OK")

    # 6. headline recompute
    agg = campaign.get("aggregate", {})
    head = campaign.get("headline", {})

    def med50(key):
        return agg.get(key, {}).get("p50_median_ms")

    recomputed_head = {}
    if med50("labA_cpu") and med50("labA_gpu"):
        recomputed_head["labA.speedup_x"] = med50("labA_cpu") / med50("labA_gpu")
    if med50("labB_pt") and med50("labB_ov"):
        recomputed_head["labB.speedup_x"] = med50("labB_pt") / med50("labB_ov")
    if med50("labC_fp32") and med50("labC_int8"):
        recomputed_head["labC.speedup_x"] = (med50("labC_fp32")
                                             / med50("labC_int8"))
    if med50("labD_b1") and med50("labD_b8"):
        recomputed_head["labD_batch.fwd_gain_x"] = \
            (8000.0 / med50("labD_b8")) / (1000.0 / med50("labD_b1"))
    for path, val in recomputed_head.items():
        sec, key = path.split(".")
        pub = head.get(sec, {}).get(key)
        if not close(val, pub, 2e-2):
            problems.append(f"headline {path}: published {pub} vs recomputed "
                            f"{val:.3f}")
        else:
            checks.append(f"headline {path} = {val:.2f}x verified")

    # cross-check aggregate per-round values against summaries we just read
    KEYMAP = {
        "labA_cpu": ("labA_cpu_vs_gpu", "pytorch_cpu"),
        "labA_gpu": ("labA_cpu_vs_gpu", "pytorch_rocm_gpu"),
        "labB_pt": ("labB_runtime_cpu", "pytorch_cpu"),
        "labB_ov": ("labB_runtime_cpu", "openvino_cpu"),
        "labC_fp32": ("labC_int8", "openvino_fp32"),
        "labC_int8": ("labC_int8", "openvino_int8"),
        "labD_res_640": ("labD_resolution", "imgsz640"),
        "labD_res_512": ("labD_resolution", "imgsz512"),
        "labD_b1": ("labD_batch", "batch1_forward"),
        "labD_b8": ("labD_batch", "batch8_forward"),
        "labD_b1_e2e": ("labD_batch", "batch1_e2e"),
        "labD_b8_e2e": ("labD_batch", "batch8_e2e"),
        "labE_serial": ("labE_pipeline", "serial"),
        "labE_pipeline": ("labE_pipeline", "async_pipeline"),
    }
    for key, r in agg.items():
        where = KEYMAP.get(key)
        if not where:
            continue
        ours = run_meta.get(where[0], {}).get(where[1])
        if not ours:
            problems.append(f"aggregate {key}: no matching runs found in "
                            f"{where}")
            continue
        p50s = [x["stats"]["p50_ms"] for x in ours]
        if not close(statistics.median(p50s), r.get("p50_median_ms", -1), 2e-2):
            problems.append(f"aggregate {key}: p50_median campaign="
                            f"{r.get('p50_median_ms')} vs recomputed "
                            f"{statistics.median(p50s):.3f}")

    # 7. Lab E parity
    labE = head.get("labE", {})
    if not labE:
        problems.append("headline.labE missing")
    else:
        if not labE.get("consistent"):
            problems.append("Lab E per-detection parity NOT consistent — "
                            "pipeline gain must not be published")
        else:
            cmp_files = sorted((root / "labE_pipeline").glob(
                "labE_pipeline_compare_*.json"))
            if not cmp_files:
                problems.append("labE_pipeline: no parity evidence files")
            else:
                last = json.loads(cmp_files[-1].read_text())
                if not last.get("consistent"):
                    problems.append("labE parity evidence file inconsistent")
                else:
                    checks.append(f"labE parity: {last['files_ok']}/"
                                  f"{last['files_compared']} files, min IoU "
                                  f"{last.get('global_min_iou')}, max dConf "
                                  f"{last.get('global_max_d_conf')}")

    # 8. mAP evidence
    maps = campaign.get("maps", {})
    for need in ("pytorch_cpu_640", "openvino_fp32_cpu_640",
                 "openvino_int8_cpu_640", "pytorch_cuda_640",
                 "pytorch_cuda_512"):
        if need not in maps:
            problems.append(f"missing mAP evaluation: {need}")
    # headline mAP values must equal the recorded evaluations
    for path, mkey in (("labB.map_pytorch", "pytorch_cpu_640"),
                       ("labB.map_openvino", "openvino_fp32_cpu_640"),
                       ("labC.map_fp32", "openvino_fp32_cpu_640"),
                       ("labC.map_int8", "openvino_int8_cpu_640"),
                       ("labD_res.map_640", "pytorch_cuda_640"),
                       ("labD_res.map_512", "pytorch_cuda_512")):
        sec, key = path.split(".")
        pub = head.get(sec, {}).get(key)
        if isinstance(pub, dict):
            pub = pub.get("map50_95")
        rec = maps.get(mkey, {}).get("map50_95")
        if pub is None:
            problems.append(f"headline {path} missing")
        elif rec is None or not close(pub, rec, 1e-6):
            problems.append(f"headline {path}: published {pub} vs maps "
                            f"{mkey}={rec}")
        else:
            checks.append(f"headline {path} = {pub:.4f} matches maps")

    # report
    try:
        campaign_rel = root.relative_to(REPO)
    except ValueError:
        campaign_rel = root
    lines = ["# Independent Benchmark Audit",
             "",
             f"Campaign: `{campaign_rel}`",
             f"Model sha256: `{model_sha}`",
             f"Files hash-verified: {n_files}",
             f"Runs recomputed: {len(checks)}",
             "",
             "## Per-run recomputation",
             ""] + [f"- {c}" for c in checks] + [
             "", "## Recomputed headline ratios", ""]
    for path, val in recomputed_head.items():
        lines.append(f"- {path} = {val:.3f}")
    lines += ["", "## Problems", ""]
    if problems:
        lines += [f"- **{p}**" for p in problems]
    else:
        lines.append("NONE — all hashes, stats, invariants, ratios and "
                     "parity checks verified independently.")
    (root / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n")

    print(f"audit: {len(checks)} runs recomputed, {n_files} files hashed")
    if problems:
        print(f"AUDIT: FAIL ({len(problems)} problems)")
        for p in problems:
            print("  -", p)
        return 1
    print("AUDIT: PASS (independent recomputation from raw bytes matches all "
          "published numbers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
