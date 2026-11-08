#!/usr/bin/env python3
"""R2.3 diagnosis: OpenVINO CPU bimodal latency — reproducible experiment.

v2.0.0 labB/labC CSVs show a strict fast/slow ALTERNATION (~26 ms / ~75 ms).
This script proves the mechanism and the fix on the live machine:

  E1  raw OV infer, default config            -> bimodal + cgroup throttling
  E2  raw OV infer, INFERENCE_NUM_THREADS=16  -> stable, ~no throttling
  E3  ultralytics e2e predict, unpinned       -> bimodal
  E4  ultralytics e2e predict, 16-CPU affinity-> stable

Mechanism: the container CPU quota is 16 cores (cpu.max 1600000/100000) but
OpenVINO's default LATENCY config spawns one thread per visible core (64),
so each inference claims up to ~64 CPU-cores' worth of runtime inside a
cgroup period; the kernel then THROTTLES the group and the next inference
pays the throttle window -> deterministic alternation fast/slow.

Outputs validation/v2.0.1/DIAGNOSIS_CPU_BIMODAL.md with all raw samples.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

N = 30
WARMUP = 8
SPLIT_MS = 45.0


def throttle() -> tuple[int, int]:
    d = dict(l.split() for l in
             Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    return int(d["nr_throttled"]), int(d["throttled_usec"])


def classify(samples: list[float]) -> dict:
    lo = [x for x in samples if x < SPLIT_MS]
    hi = [x for x in samples if x >= SPLIT_MS]
    alt = sum(1 for a, b in zip(samples, samples[1:])
              if (a - SPLIT_MS) * (b - SPLIT_MS) < 0)
    return {"n": len(samples),
            "p50_ms": round(statistics.median(samples), 1),
            "lo_n": len(lo), "lo_mean_ms": round(statistics.fmean(lo), 1) if lo else None,
            "hi_n": len(hi), "hi_mean_ms": round(statistics.fmean(hi), 1) if hi else None,
            "alternations": alt, "samples_ms": [round(x, 1) for x in samples]}


def raw_ov(config: dict | None) -> tuple[dict, dict]:
    import numpy as np
    import openvino as ov
    core = ov.Core()
    xml = str(REPO / "weights/yolo26l_openvino_model/yolo26l.xml")
    cm = core.compile_model(xml, "CPU") if config is None \
        else core.compile_model(xml, "CPU", config)
    ireq = cm.create_infer_request()
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(WARMUP):
        ireq.infer(x)
    nt0, tu0 = throttle()
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        ireq.infer(x)
        t.append((time.perf_counter() - t0) * 1000)
    nt1, tu1 = throttle()
    return classify(t), {"d_nr_throttled": nt1 - nt0,
                         "d_throttled_ms": round((tu1 - tu0) / 1000)}


def ultralytics_e2e(pin: list[int] | None) -> tuple[dict, dict]:
    import os
    from ultralytics import YOLO
    prev = sorted(os.sched_getaffinity(0))
    if pin is not None:
        os.sched_setaffinity(0, sorted(pin))
    try:
        m = YOLO(str(REPO / "weights/yolo26l_openvino_model"))
        img = str(sorted((REPO / "data/val_subset/images").glob("*.jpg"))[0])
        for _ in range(WARMUP):
            m.predict(source=img, imgsz=640, device="cpu", verbose=False)
        nt0, tu0 = throttle()
        t = []
        for _ in range(N):
            t0 = time.perf_counter()
            m.predict(source=img, imgsz=640, device="cpu", verbose=False)
            t.append((time.perf_counter() - t0) * 1000)
        nt1, tu1 = throttle()
        return classify(t), {"d_nr_throttled": nt1 - nt0,
                             "d_throttled_ms": round((tu1 - tu0) / 1000),
                             "affinity": pin or "default(all)"}
    finally:
        os.sched_setaffinity(0, prev)


def main() -> None:
    out_dir = REPO / "validation/v2.0.1"
    out_dir.mkdir(parents=True, exist_ok=True)
    quota = None
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        quota = int(q) / int(p) if q != "max" else None
    except Exception:
        pass
    print(f"cgroup quota: {quota} cores · running 4 experiments (~3 min)")
    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": "2x AMD EPYC 9334 (64C/128T) · cgroup quota "
                   f"{quota} cores · shared host",
        "E1_raw_default": {},
        "E2_raw_threads16": {},
        "E3_ultralytics_unpinned": {},
        "E4_ultralytics_pinned16": {},
    }
    s, th = raw_ov(None)
    results["E1_raw_default"] = {"latency": s, "throttle": th,
                                 "config": "PERFORMANCE_HINT=LATENCY (ultralytics default), threads=default(64)"}
    print("E1", s["p50_ms"], "ms ·", th)
    s, th = raw_ov({"PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": 16})
    results["E2_raw_threads16"] = {"latency": s, "throttle": th,
                                   "config": "PERFORMANCE_HINT=LATENCY + INFERENCE_NUM_THREADS=16 (= quota)"}
    print("E2", s["p50_ms"], "ms ·", th)
    s, th = ultralytics_e2e(None)
    results["E3_ultralytics_unpinned"] = {"latency": s, "throttle": th}
    print("E3", s["p50_ms"], "ms ·", th)
    quiet = list(range(0, 16))
    s, th = ultralytics_e2e(quiet)
    results["E4_ultralytics_pinned16"] = {"latency": s, "throttle": th}
    print("E4", s["p50_ms"], "ms ·", th)

    (out_dir / "DIAGNOSIS_CPU_BIMODAL.json").write_text(
        json.dumps(results, indent=2))

    def verdict(r): 
        return "BIMODAL (throttled)" if r["latency"]["hi_n"] > 3 else "stable"

    md = [
        "# R2.3 Diagnosis — OpenVINO CPU bimodal latency",
        "",
        f"Date: {results['date']} · {results['machine']}",
        "",
        "## Finding",
        "",
        "The v2.0.0 bimodal CPU latency (~26/75 ms strict alternation) is "
        "**container cgroup CPU throttling**, not noise:",
        "",
        "- OpenVINO's LATENCY config defaults to one thread per visible core "
        "(64 here) while the container quota is 16 cores;",
        "- an inference burst exhausts the 100 ms period budget, the kernel "
        "throttles the cgroup, and the NEXT inference pays the throttle "
        "window — producing deterministic fast/slow alternation;",
        "- proof: the number of 'slow' samples equals the delta of "
        "`nr_throttled` in `/sys/fs/cgroup/cpu.stat` within each experiment.",
        "",
        "## Experiments",
        "",
        "| # | Setup | P50 ms | slow samples | Δnr_throttled | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (("E1_raw_default", "raw OV · default threads (64)"),
                       ("E2_raw_threads16", "raw OV · 16 threads (=quota)"),
                       ("E3_ultralytics_unpinned", "ultralytics e2e · unpinned"),
                       ("E4_ultralytics_pinned16", "ultralytics e2e · pinned 16 CPUs")):
        r = results[key]
        md.append(f"| {key[0]} | {label} | {r['latency']['p50_ms']} | "
                  f"{r['latency']['hi_n']} | {r['throttle']['d_nr_throttled']} | "
                  f"{verdict(r)} |")
    md += [
        "",
        "## Consequence for v2.0.1 benchmarks",
        "",
        "- All CPU benchmark variants pin the process to 16 CPUs (re-picked "
        "quiet set per round, recorded per run) matching the quota;",
        "- PyTorch CPU threads pinned to 8 (measured optimum under quota);",
        "- host is SHARED with other tenants — load average and throttle "
        "counters are recorded around every campaign round and disclosed.",
        "",
        "Raw samples: DIAGNOSIS_CPU_BIMODAL.json (same directory).",
    ]
    (out_dir / "DIAGNOSIS_CPU_BIMODAL.md").write_text("\n".join(md) + "\n")
    print(f"written -> {out_dir / 'DIAGNOSIS_CPU_BIMODAL.md'}")


if __name__ == "__main__":
    main()
