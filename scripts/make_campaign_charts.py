#!/usr/bin/env python3
"""Generate the six slide charts from a formal v2.0.1 campaign (R2.7).

Reads CAMPAIGN.json (median-of-rounds aggregates + per-round runs) and
produces assets/slides/01..06 PNG+SVG with full evidence footnotes:
campaign id, model sha, hardware, timing scope, rounds, evidence paths.

Usage:
    python scripts/make_campaign_charts.py --campaign validation/v2.0.1/campaigns/<id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from course import charts   # noqa: E402


def load(croot: Path, key: str) -> dict:
    """Load the LAST round's summary for a run key (footnote shows all)."""
    c = json.loads((croot / "CAMPAIGN.json").read_text())
    rid = c["run_ids"][key][-1]
    p = next(croot.glob(f"*/{rid}_summary.json"))
    return json.loads(p.read_text())


def foot(croot: Path, c: dict, keys: list[str], extra: str = "") -> str:
    ids = " · ".join(f"{c['run_ids'][k][-1]}" for k in keys)
    m = c["model"]
    rounds = c["rounds"]
    return (f"Campaign {c['campaign_id']} · {rounds} rounds x {c['iters_per_variant']} "
            f"iters (median of round P50s) · YOLO26l sha {m['sha256'][:12]}… · "
            f"AMD EPYC 9334 + W7900 (gfx1100) · shared host, load recorded · "
            f"runs: {ids} · evidence: {croot.name}/ · SHA256SUMS verified "
            f"{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    croot = Path(args.campaign)
    if not croot.is_absolute():
        croot = REPO / croot
    c = json.loads((croot / "CAMPAIGN.json").read_text())
    agg, head, maps = c["aggregate"], c["headline"], c["maps"]

    cpu = load(croot, "labA_cpu")
    gpu = load(croot, "labA_gpu")
    charts.chart_cpu_vs_gpu(
        {"stats": {"p50_ms": head["labA"]["cpu_p50_ms"],
                   "p95_ms": agg["labA_cpu"]["p95_median_ms"]}},
        {"stats": {"p50_ms": head["labA"]["gpu_p50_ms"],
                   "p95_ms": agg["labA_gpu"]["p95_median_ms"]}},
        foot=foot(croot, c, ["labA_cpu", "labA_gpu"],
                  " · scope: end-to-end per image (pre+infer+post)"))

    pt = load(croot, "labB_pt")
    ov = load(croot, "labB_ov")
    charts.chart_pytorch_vs_openvino(
        {"stats": {"p50_ms": head["labB"]["pt_p50_ms"], "min_ms": 0,
                   "p95_ms": agg["labB_pt"]["p95_median_ms"]}},
        {"stats": {"p50_ms": head["labB"]["ov_p50_ms"], "min_ms": 0,
                   "p95_ms": agg["labB_ov"]["p95_median_ms"]}},
        maps["pytorch_cpu_640"]["map50_95"],
        maps["openvino_fp32_cpu_640"]["map50_95"],
        foot=foot(croot, c, ["labB_pt", "labB_ov"],
                  " · scope: e2e per image, same CPU, A/B interleaved order"))

    from course.quantization import model_dir_size_mb
    from course.model import openvino_fp32_dir, openvino_int8_dir
    fp32_dir = openvino_fp32_dir()
    int8_dir = openvino_int8_dir()
    charts.chart_fp32_vs_int8(
        {"stats": {"p50_ms": head["labC"]["fp32_p50_ms"]}},
        {"stats": {"p50_ms": head["labC"]["int8_p50_ms"]}},
        model_dir_size_mb(fp32_dir), model_dir_size_mb(int8_dir),
        maps["openvino_fp32_cpu_640"]["map50_95"],
        maps["openvino_int8_cpu_640"]["map50_95"],
        foot=foot(croot, c, ["labC_fp32", "labC_int8"],
                  " · quantization REBUILT this campaign (NNCF MIXED, 150 real "
                  "train2017 calib images, identity in CACHE_KEY.json)"),
        subtitle=("YOLO26l · OpenVINO · NNCF MIXED PTQ · 150 real COCO "
                  "train2017 calibration images (this campaign's fresh build)"))

    s640 = load(croot, "labD_res_640")
    s512 = load(croot, "labD_res_512")
    charts.chart_resolution(
        640, 512,
        {"p50_ms": head["labD_res"]["i640_p50_ms"]},
        {"p50_ms": head["labD_res"]["i512_p50_ms"]},
        maps["pytorch_cuda_640"]["map50_95"],
        maps["pytorch_cuda_512"]["map50_95"],
        s640.get("derived", {}).get("vram_peak_gb", 0),
        s512.get("derived", {}).get("vram_peak_gb", 0),
        s512,
        foot=foot(croot, c, ["labD_res_640", "labD_res_512"],
                  " · scope: e2e per image on W7900"))

    b1 = head["labD_batch"]["b1_fwd_p50_ms"]
    b8 = head["labD_batch"]["b8_fwd_p50_ms"]
    e1 = head["labD_batch"]["b1_e2e_p50_ms"]
    e8 = head["labD_batch"]["b8_e2e_p50_ms"]
    charts.chart_batch(
        1, 8, 1000.0 / b1, 8000.0 / b8,
        f"{b1:.1f} ms/fwd", f"{b8:.1f} ms/batch",
        0, 0, load(croot, "labD_b8"),
        foot=foot(croot, c, ["labD_b1", "labD_b8", "labD_b1_e2e", "labD_b8_e2e"],
                  " · forward-only throughput is NOT end-to-end"),
        e2e_note=(f"YOLO26l · W7900 · forward-only: {b1:.1f} ms/img (b1) vs "
                  f"{b8:.1f} ms/batch ÷ 8 = {b8/8:.1f} ms/img-equiv (b8) — "
                  f"NOT request latency · end-to-end: {e1:.1f} ms (b1) vs "
                  f"{e8:.1f} ms per 8-image request (b8)"))

    le = head["labE"]
    ser_recs = sorted((croot / "labE_pipeline").glob(
        "labE_pipeline_serial_*_records.json"))[-1]
    pipe_recs = sorted((croot / "labE_pipeline").glob(
        "labE_pipeline_async_pipeline_*_records.json"))[-1]
    serial_rec = json.loads(ser_recs.read_text())
    pipe_rec = json.loads(pipe_recs.read_text())
    charts.chart_pipeline(
        serial_rec, pipe_rec,
        foot=foot(croot, c, ["labE_serial", "labE_pipeline"],
                  " · per-detection parity: class+box IoU>=0.95+conf±0.02, "
                  f"{le['files_ok']}/{le['files_compared']} files OK, min IoU "
                  f"{le['global_min_iou']} · serial e2e has no queue; "
                  "pipeline e2e is burst submit->result incl. queue wait"))
    print("charts regenerated from campaign", c["campaign_id"])


if __name__ == "__main__":
    main()
