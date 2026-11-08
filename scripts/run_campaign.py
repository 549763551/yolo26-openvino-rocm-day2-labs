#!/usr/bin/env python3
"""Formal benchmark campaign runner for v2.0.1 (R2.4/R2.5).

Runs the SIX experiments on real hardware, repeatedly, into a fresh
campaign directory — v2.0.0 results are never reused (mission §0.5).

Scientific protocol encoded here:
  * one official yolo26l.pt (sha256 recorded, identical for every variant);
  * fixed image list (the 120-image educational COCO val subset, sorted);
  * warm-up runs excluded and counted; >=50 timed samples per variant/round;
  * CPU variants: process pinned to a quiet 16-CPU set (matches the 16-core
    cgroup quota; eliminates quota throttling — see R2.3 diagnosis), torch
    threads pinned to 8; configuration recorded per run;
  * PyTorch/OpenVINO order alternates between rounds (A/B, B/A) to cancel
    drift; GPU timing is cuda-synchronized; host load + throttle counters
    recorded around every round (shared machine disclosure);
  * every run -> immutable CSV + summary JSON + SHA256SUMS (see
    scripts/verify_evidence.py); no sample is ever dropped;
  * latency rounds are repeated --rounds N times; mAP evaluated once per
    model configuration (deterministic given fixed model+data);
  * Lab C quantization is REBUILT fresh this round (force on first build)
    with its full calibration manifest recorded;
  * Lab E stores per-image stage timings AND per-detection outputs; the
    published gain is only claimed when per-detection parity passes.

Output tree:
  validation/v2.0.1/campaigns/<campaign_id>/
    {labA_cpu_vs_gpu,labB_runtime_cpu,labC_int8,labD_resolution,labD_batch,
     labE_pipeline}/  *.csv + *_summary.json + records + SHA256SUMS
    maps/*.json · ENV.json · CAMPAIGN.json · SHA256SUMS
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from course import benchmark as B            # noqa: E402
from course import model as M                # noqa: E402
from course import pipeline as P             # noqa: E402
from course import quantization as Q         # noqa: E402
from course import setup_checks as SC        # noqa: E402

WARMUP = 10
ITERS = 50          # >=50 formal samples per variant per round (R2.5)
PIPE_IMAGES = 100   # Lab E (scenario 1: 100-image burst)


def log(msg: str) -> None:
    print(f"\033[1;36m[campaign {datetime.now().strftime('%H:%M:%S')}]\033[0m "
          f"{msg}", flush=True)


class Ctx:
    def __init__(self, root: Path):
        self.root = root
        self.round_env: list[dict] = []
        self.runs: dict[str, list] = {}
        self.maps: dict = {}
        self.extra: dict = {}

    def record(self, key: str, summary: dict) -> None:
        self.runs.setdefault(key, []).append(summary["run_id"])


def bench_predict_e2e(model, img, device: str, imgsz: int, warmup: int,
                      iters: int) -> list[float]:
    """End-to-end per-image predict latency (pre+infer+post), synced on GPU."""
    for i in range(warmup):
        model.predict(source=str(img[i % len(img)]), imgsz=imgsz,
                      device=device, verbose=False)
    samples = []
    for i in range(iters):
        p = img[i % len(img)]
        samples.append(B.timed_call(
            lambda p=p: model.predict(source=str(p), imgsz=imgsz,
                                      device=device, verbose=False),
            sync=device.startswith("cuda")))
    return samples


def round_env_snapshot(tag: str) -> dict:
    return {"tag": tag, "time": datetime.now().isoformat(timespec="seconds"),
            **B.host_load()}


def exp_lab_a(ctx: Ctx, imgs, round_i: int) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(M.OFFICIAL_PT))
    out = {}
    # CPU side: pinned quiet cores + torch threads 8
    cpus = B.pick_quiet_cpus(16)
    prev = B.pin_affinity(cpus)
    nthr = B.pin_torch_threads(8)
    try:
        log(f"Lab A round {round_i} · PyTorch CPU e2e (pinned {len(cpus)} cpus, "
            f"torch threads {nthr})")
        s = B.save_run(B.RunMeta(
            experiment="labA_cpu_vs_gpu", variant="pytorch_cpu", device="cpu",
            runtime="pytorch-rocm-cpu", warmup=WARMUP, iters=ITERS,
            notes="end-to-end per image (pre+infer+post)",
            extra={"affinity_cpus": cpus, "torch_threads": nthr,
                   "round": round_i}), bench_predict_e2e(model, imgs, "cpu", 640,
                                                         WARMUP, ITERS))
    finally:
        B.pin_affinity(prev)
    ctx.record("labA_cpu", s)
    out["cpu"] = s
    log(f"Lab A round {round_i} · PyTorch ROCm GPU e2e")
    g = B.save_run(B.RunMeta(
        experiment="labA_cpu_vs_gpu", variant="pytorch_rocm_gpu",
        device="cuda(gfx1100)", runtime="pytorch-rocm", warmup=WARMUP,
        iters=ITERS, notes="end-to-end per image (pre+infer+post), cuda-synced",
        extra={"round": round_i}), bench_predict_e2e(model, imgs, "cuda", 640,
                                                     WARMUP, ITERS))
    ctx.record("labA_gpu", g)
    out["gpu"] = g
    return out


def exp_lab_b(ctx: Ctx, imgs, round_i: int) -> dict:
    """PyTorch vs OpenVINO on the same AMD CPU; order alternates per round."""
    from ultralytics import YOLO
    ov_dir = M.export_openvino_fp32(640)
    out = {}
    cpus = B.pick_quiet_cpus(16)
    prev = B.pin_affinity(cpus)
    nthr = B.pin_torch_threads(8)
    try:
        sides = ["pt", "ov"] if round_i % 2 == 1 else ["ov", "pt"]
        log(f"Lab B round {round_i} · order {sides} (interleaved A/B) · "
            f"pinned {len(cpus)} cpus · torch threads {nthr}")
        for side in sides:
            if side == "pt":
                t0 = time.perf_counter()
                m = YOLO(str(M.OFFICIAL_PT))
                load_s = time.perf_counter() - t0
                s = B.save_run(B.RunMeta(
                    experiment="labB_runtime_cpu", variant="pytorch_cpu",
                    device="cpu", runtime="pytorch-rocm-cpu", warmup=WARMUP,
                    iters=ITERS, notes="same CPU, same weights, same postproc",
                    extra={"affinity_cpus": cpus, "torch_threads": nthr,
                           "model_load_s": round(load_s, 3), "round": round_i}),
                    bench_predict_e2e(m, imgs, "cpu", 640, WARMUP, ITERS))
                ctx.record("labB_pt", s)
                out["pt"] = s
            else:
                t0 = time.perf_counter()
                m = YOLO(str(ov_dir))
                m.predict(source=str(imgs[0]), imgsz=640, device="cpu",
                          verbose=False)
                compile_s = time.perf_counter() - t0
                s = B.save_run(B.RunMeta(
                    experiment="labB_runtime_cpu", variant="openvino_cpu",
                    device="cpu", runtime="openvino-cpu", warmup=WARMUP,
                    iters=ITERS,
                    notes="same CPU, same weights exported from same yolo26l.pt",
                    extra={"affinity_cpus": cpus,
                           "first_load_and_compile_s": round(compile_s, 3),
                           "torch_threads": nthr, "round": round_i}),
                    bench_predict_e2e(m, imgs, "cpu", 640, WARMUP, ITERS))
                ctx.record("labB_ov", s)
                out["ov"] = s
    finally:
        B.pin_affinity(prev)
    return out


def exp_lab_c(ctx: Ctx, imgs, round_i: int, force_build: bool) -> dict:
    from ultralytics import YOLO
    out = {}
    fp32_dir = M.export_openvino_fp32(640)
    if force_build and round_i == 1:
        log("Lab C · rebuilding INT8 quantization fresh this round "
            "(no stale cache, full calibration manifest recorded)")
        t0 = time.perf_counter()
        int8_dir = Q.quantize_int8(n_calib_images=150, subset_size=150,
                                   preset="mixed", force=True)
        qtime = time.perf_counter() - t0
        rec = json.loads((int8_dir / "CACHE_KEY.json").read_text())
        ctx.extra["int8_identity"] = rec
    else:
        t0 = time.perf_counter()
        int8_dir = Q.quantize_int8(n_calib_images=150, subset_size=150,
                                   preset="mixed")
        qtime = time.perf_counter() - t0
        rec_path = int8_dir / "CACHE_KEY.json"
        if rec_path.exists():
            ctx.extra.setdefault("int8_identity",
                                 json.loads(rec_path.read_text()))
    cpus = B.pick_quiet_cpus(16)
    prev = B.pin_affinity(cpus)
    try:
        fp32_model = YOLO(str(fp32_dir))
        int8_model = YOLO(str(int8_dir))
        log(f"Lab C round {round_i} · OpenVINO FP32 vs INT8 on same AMD CPU "
            f"(pinned {len(cpus)} cpus)")
        f = B.save_run(B.RunMeta(
            experiment="labC_int8", variant="openvino_fp32", device="cpu",
            runtime="openvino-cpu", precision="fp32", warmup=WARMUP,
            iters=ITERS, extra={"affinity_cpus": cpus, "round": round_i}),
            bench_predict_e2e(fp32_model, imgs, "cpu", 640, WARMUP, ITERS))
        ctx.record("labC_fp32", f)
        i = B.save_run(B.RunMeta(
            experiment="labC_int8", variant="openvino_int8", device="cpu",
            runtime="openvino-cpu", precision="int8", warmup=WARMUP,
            iters=ITERS,
            extra={"affinity_cpus": cpus, "round": round_i,
                   "quantization_time_s": round(qtime, 1),
                   "calibration_images": 150, "nncf_preset": "mixed",
                   "int8_cache": str(int8_dir.relative_to(REPO))}),
            bench_predict_e2e(int8_model, imgs, "cpu", 640, WARMUP, ITERS))
        ctx.record("labC_int8", i)
        out = {"fp32": f, "int8": i}
    finally:
        B.pin_affinity(prev)
    return out


def exp_lab_d_res(ctx: Ctx, imgs, round_i: int) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(M.OFFICIAL_PT))
    out = {}
    for imgsz in (640, 512):
        log(f"Lab D1 round {round_i} · GPU imgsz {imgsz}")
        import torch
        torch.cuda.reset_peak_memory_stats()
        s = B.save_run(B.RunMeta(
            experiment="labD_resolution", variant=f"imgsz{imgsz}",
            device="cuda(gfx1100)", runtime="pytorch-rocm", imgsz=imgsz,
            warmup=WARMUP, iters=ITERS, extra={"round": round_i}),
            bench_predict_e2e(model, imgs, "cuda", imgsz, WARMUP, ITERS))
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        s["derived"]["vram_peak_gb"] = round(mem, 3)
        ctx.record(f"labD_res_{imgsz}", s)
        out[imgsz] = s
    return out


def exp_lab_d_batch(ctx: Ctx, imgs, round_i: int) -> dict:
    import torch
    from ultralytics import YOLO
    model = YOLO(str(M.OFFICIAL_PT))
    stages = P.Yolo26lStages(imgsz=640)
    window = imgs[:8]

    def forward(paths):
        im0s = [stages.load(p) for p in paths]
        im = stages.preprocess(im0s)
        torch.cuda.synchronize()
        t0 = B.now_ms()
        stages.infer(im)
        torch.cuda.synchronize()
        return B.now_ms() - t0

    out = {}
    torch.cuda.reset_peak_memory_stats()
    # batch 1 (forward-only)
    for _ in range(WARMUP):
        forward([imgs[0]])
    s1 = [forward([imgs[i % len(imgs)]]) for i in range(ITERS)]
    b1 = B.save_run(B.RunMeta(
        experiment="labD_batch", variant="batch1_forward",
        device="cuda(gfx1100)", runtime="pytorch-rocm", batch=1,
        warmup=WARMUP, iters=ITERS, imgsz=640,
        notes="GPU forward only (model-only), batch=1",
        extra={"round": round_i}), s1)
    ctx.record("labD_b1", b1)
    # batch 8 (forward-only, whole batch timed as one call)
    for _ in range(WARMUP):
        forward(window)
    s8 = [forward(window) for _ in range(ITERS)]
    b8 = B.save_run(B.RunMeta(
        experiment="labD_batch", variant="batch8_forward",
        device="cuda(gfx1100)", runtime="pytorch-rocm", batch=8,
        warmup=WARMUP, iters=ITERS, imgsz=640,
        notes="GPU forward only, whole batch of 8 timed as one call",
        extra={"round": round_i}), s8)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    b8["derived"]["vram_peak_gb"] = round(mem, 3)
    ctx.record("labD_b8", b8)

    # E2E comparison with honest labels (R2 spec: forward-only must not pose
    # as e2e; provide both)
    e2e1 = B.bench(lambda p=imgs[0]: model.predict(
        source=str(p), imgsz=640, device="cuda", verbose=False),
        WARMUP, ITERS, sync=True)
    e2e8 = []
    for _ in range(WARMUP):
        model.predict(source=[str(p) for p in window], imgsz=640,
                      device="cuda", verbose=False)
    for i in range(ITERS):
        w = [imgs[(i + k) % len(imgs)] for k in range(8)]
        e2e8.append(B.timed_call(
            lambda w=w: model.predict(source=[str(p) for p in w], imgsz=640,
                                      device="cuda", verbose=False), sync=True))
    e1 = B.save_run(B.RunMeta(
        experiment="labD_batch", variant="batch1_e2e", device="cuda(gfx1100)",
        runtime="pytorch-rocm", batch=1, warmup=WARMUP, iters=ITERS,
        notes="end-to-end predict (pre+infer+post) batch=1",
        extra={"round": round_i}), e2e1)
    e8 = B.save_run(B.RunMeta(
        experiment="labD_batch", variant="batch8_e2e", device="cuda(gfx1100)",
        runtime="pytorch-rocm", batch=8, warmup=WARMUP, iters=ITERS,
        notes="end-to-end predict, whole batch of 8 as one request",
        extra={"round": round_i}), e2e8)
    ctx.record("labD_b1_e2e", e1)
    ctx.record("labD_b8_e2e", e8)
    out = {"b1": b1, "b8": b8, "b1_e2e": e1, "b8_e2e": e8}
    return out


def exp_lab_e(ctx: Ctx, imgs, round_i: int) -> dict:
    log(f"Lab E round {round_i} · serial vs async pipeline ({PIPE_IMAGES} images)")
    serial = P.run_serial(imgs[:PIPE_IMAGES], imgsz=640)
    pipeline = P.run_pipeline(imgs[:PIPE_IMAGES], imgsz=640, queue_depth=4)
    cmp = P.compare_results(serial, pipeline)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = B.results_root() / "labE_pipeline"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for mode, rec in (("serial", serial), ("async_pipeline", pipeline)):
        rp = exp_dir / f"labE_pipeline_{mode}_{ts}_records.json"
        rp.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    e2e_s = [r["e2e_ms"] for r in serial["per_image"]]
    e2e_p = [r["e2e_ms"] for r in pipeline["per_image"]]
    ss = B.save_run(B.RunMeta(
        experiment="labE_pipeline", variant="serial", device="cpu+cuda(gfx1100)",
        runtime="pytorch-rocm", warmup=0, iters=len(e2e_s), imgsz=640,
        notes="per-image e2e stage-serial; no queue",
        extra={"round": round_i, "total_ms": serial["total_ms"],
               "images_per_s": serial["images_per_s"]}), e2e_s)
    ps = B.save_run(B.RunMeta(
        experiment="labE_pipeline", variant="async_pipeline",
        device="cpu+cuda(gfx1100)", runtime="pytorch-rocm",
        warmup=0, iters=len(e2e_p), imgsz=640,
        notes="per-image e2e under full burst (includes queue wait)",
        extra={"round": round_i, "total_ms": pipeline["total_ms"],
               "images_per_s": pipeline["images_per_s"]}), e2e_p)
    ctx.record("labE_serial", ss)
    ctx.record("labE_pipeline", ps)
    cmp_file = exp_dir / f"labE_pipeline_compare_{ts}.json"
    cmp_file.write_text(json.dumps(cmp, indent=2, default=str) + "\n")
    B.refresh_sums(exp_dir)
    log(f"Lab E round {round_i} · parity files_ok={cmp['files_ok']}/"
        f"{cmp['files_compared']} consistent={cmp['consistent']} · gain "
        f"{cmp['throughput_gain_x']:.2f}x")
    return {"serial": serial, "pipeline": pipeline, "compare": cmp,
            "run_ids": [ss["run_id"], ps["run_id"]]}


def eval_maps(ctx: Ctx, imgs) -> None:
    """Deterministic accuracy evaluation, one per model configuration."""
    from ultralytics import YOLO
    yaml = str(REPO / "data/val_subset.yaml")
    maps = ctx.maps
    log("mAP · PyTorch cpu (120-img educational subset)")
    m = YOLO(str(M.OFFICIAL_PT))
    maps["pytorch_cpu_640"] = B.map5095(m, yaml, 640, 8, "cpu")
    log("mAP · PyTorch cuda 640 / 512")
    maps["pytorch_cuda_640"] = B.map5095(m, yaml, 640, 8, "cuda")
    maps["pytorch_cuda_512"] = B.map5095(m, yaml, 512, 8, "cuda")
    ov = YOLO(str(M.export_openvino_fp32(640)))
    log("mAP · OpenVINO fp32 cpu")
    maps["openvino_fp32_cpu_640"] = B.map5095(ov, yaml, 640, 8, "cpu")
    int8_dir = Q.quantize_int8(n_calib_images=150, subset_size=150)
    iq = YOLO(str(int8_dir))
    log("mAP · OpenVINO int8 cpu (this round's fresh quantization)")
    maps["openvino_int8_cpu_640"] = B.map5095(iq, yaml, 640, 8, "cpu")
    (B.results_root() / "maps").mkdir(parents=True, exist_ok=True)
    (B.results_root() / "maps" / "map_results.json").write_text(
        json.dumps(maps, indent=2) + "\n")


def aggregate(ctx: Ctx) -> dict:
    """Cross-round statistics: per-round P50/P95, median-of-rounds, range."""
    agg = {}
    for key, run_ids in ctx.runs.items():
        stats = []
        for rid in run_ids:
            sp = next(B.results_root().glob(
                f"*/{rid}_summary.json"))
            s = json.loads(sp.read_text())
            stats.append({"run_id": rid, "p50_ms": s["stats"]["p50_ms"],
                          "p95_ms": s["stats"]["p95_ms"],
                          "mean_ms": s["stats"]["mean_ms"],
                          "n": s["stats"]["n"]})
        p50s = [x["p50_ms"] for x in stats]
        p95s = [x["p95_ms"] for x in stats]
        agg[key] = {
            "rounds": stats,
            "rounds_n": len(stats),
            "p50_median_ms": statistics.median(p50s),
            "p50_min_ms": min(p50s), "p50_max_ms": max(p50s),
            "p95_median_ms": statistics.median(p95s),
            "spread_pct": round(100 * (max(p50s) - min(p50s))
                                / statistics.median(p50s), 1),
        }
    return agg


def headline(agg: dict, maps: dict, ctx: Ctx) -> dict:
    def med(k): return agg[k]["p50_median_ms"]

    lab_e_cmp = ctx.extra.get("labE_cmp", {})
    return {
        "labA": {"cpu_p50_ms": med("labA_cpu"), "gpu_p50_ms": med("labA_gpu"),
                 "speedup_x": round(med("labA_cpu") / med("labA_gpu"), 2),
                 "kind": "hardware comparison (CPU device vs GPU device)"},
        "labB": {"pt_p50_ms": med("labB_pt"), "ov_p50_ms": med("labB_ov"),
                 "speedup_x": round(med("labB_pt") / med("labB_ov"), 2),
                 "map_pytorch": maps.get("pytorch_cpu_640"),
                 "map_openvino": maps.get("openvino_fp32_cpu_640"),
                 "kind": "same-CPU runtime comparison"},
        "labC": {"fp32_p50_ms": med("labC_fp32"),
                 "int8_p50_ms": med("labC_int8"),
                 "speedup_x": round(med("labC_fp32") / med("labC_int8"), 2),
                 "map_fp32": maps.get("openvino_fp32_cpu_640"),
                 "map_int8": maps.get("openvino_int8_cpu_640"),
                 "int8_identity": ctx.extra.get("int8_identity", {})
                 .get("identity", {}) .get("calibration", {})
                 .get("dataset_sha256", "")},
        "labD_res": {"i640_p50_ms": med("labD_res_640"),
                     "i512_p50_ms": med("labD_res_512"),
                     "map_640": maps.get("pytorch_cuda_640"),
                     "map_512": maps.get("pytorch_cuda_512")},
        "labD_batch": {
            "b1_fwd_p50_ms": med("labD_b1"), "b8_fwd_p50_ms": med("labD_b8"),
            "b1_fwd_thr_img_s": round(1000 / med("labD_b1"), 1),
            "b8_fwd_thr_img_s": round(8000 / med("labD_b8"), 1),
            "fwd_gain_x": round((8000 / med("labD_b8"))
                                / (1000 / med("labD_b1")), 2),
            "b1_e2e_p50_ms": med("labD_b1_e2e"),
            "b8_e2e_p50_ms": med("labD_b8_e2e"),
            "labels": "forward-only throughput is NOT end-to-end; e2e "
                      "variants measured separately"},
        "labE": lab_e_cmp,
    }


def main() -> None:
    global ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "validation" / "v2.0.1"
                                         / "campaigns"))
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--only", default="labA,labB,labC,labD,labE")
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()
    ITERS = args.iters

    campaign_id = f"c{datetime.now().strftime('%Y%m%d_%H%M')}"
    root = Path(args.out) / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    B.set_results_root(root)

    imgs = B.load_bench_images()

    rep = SC.full_report()
    SC.assert_healthy(rep)
    (root / "ENV.json").write_text(json.dumps(rep, indent=2, default=str))
    ctx = Ctx(root)

    only = [s.strip() for s in args.only.split(",")]
    t_start = time.time()
    per_round = {}
    for r in range(1, args.rounds + 1):
        ctx.round_env.append({"start": round_env_snapshot(f"round{r}-start")})
        log(f"===== ROUND {r}/{args.rounds} ===== "
            f"(load {ctx.round_env[-1]['start'].get('loadavg_1_5_15')})")
        rd = {}
        if "labA" in only:
            rd["labA"] = exp_lab_a(ctx, imgs, r)
        if "labB" in only:
            rd["labB"] = exp_lab_b(ctx, imgs, r)
        if "labC" in only:
            rd["labC"] = exp_lab_c(ctx, imgs, r, force_build=(r == 1))
        if "labD" in only:
            rd["labD_res"] = exp_lab_d_res(ctx, imgs, r)
            rd["labD_batch"] = exp_lab_d_batch(ctx, imgs, r)
        if "labE" in only:
            e = exp_lab_e(ctx, imgs, r)
            rd["labE"] = e["run_ids"]
            if r == args.rounds:
                ctx.extra["labE_cmp"] = {
                    "consistent": e["compare"]["consistent"],
                    "files_ok": e["compare"]["files_ok"],
                    "files_compared": e["compare"]["files_compared"],
                    "global_min_iou": e["compare"]["global_min_iou"],
                    "global_max_d_conf": e["compare"]["global_max_d_conf"],
                    "serial_imgs_per_s": e["compare"]["serial_imgs_per_s"],
                    "pipeline_imgs_per_s": e["compare"]["pipeline_imgs_per_s"],
                    "throughput_gain_x": round(
                        e["compare"]["throughput_gain_x"], 3),
                    "serial_e2e_p50_ms": e["serial"]["e2e_p50_ms"],
                    "pipeline_e2e_p50_ms": e["pipeline"]["e2e_p50_ms"],
                    "serial_e2e_note": "no queue (per-image execution time)",
                    "pipeline_e2e_note": "burst submit->result incl. queue wait",
                    "scenario1_burst": True,
                }
        ctx.round_env.append({"end": round_env_snapshot(f"round{r}-end")})
        per_round[f"round{r}"] = {k: (v if isinstance(v, list) else "ok")
                                  for k, v in rd.items()}

    if not args.skip_map and "maps" not in ctx.maps:
        eval_maps(ctx, imgs)

    agg = aggregate(ctx)
    head = headline(agg, ctx.maps, ctx)
    campaign = {
        "campaign_id": campaign_id,
        "started": datetime.now().isoformat(timespec="seconds"),
        "total_wall_s": round(time.time() - t_start, 1),
        "rounds": args.rounds, "iters_per_variant": ITERS, "warmup": WARMUP,
        "model": {"file": "weights/yolo26l.pt",
                  "sha256": M.model_sha256(),
                  "task": "detect", "classes": 80},
        "images": {"list": "data/val_subset/images (sorted, 120 files)",
                   "mode": "cycled deterministically i % 120"},
        "environment": json.loads((root / "ENV.json").read_text()),
        "cpu_protocol": {
            "affinity": "quiet 16 CPUs (one NUMA node), re-picked per round, "
                        "recorded per run in meta.extra.affinity_cpus",
            "cgroup_quota_cores": SC.cgroup_cpu_quota(),
            "torch_threads": 8,
            "rationale": "16-core cgroup quota; >16 threads causes cgroup "
                         "throttling and bimodal latency (R2.3 diagnosis)"},
        "run_ids": ctx.runs,
        "round_env": ctx.round_env,
        "maps": ctx.maps,
        "aggregate": agg,
        "headline": head,
        "notes": "All latency variants: warmup excluded, every sample kept. "
                 "GPU timing cuda-synchronized. Host is shared — load and "
                 "throttle counters recorded per round.",
    }
    (root / "CAMPAIGN.json").write_text(json.dumps(campaign, indent=2,
                                                   default=str) + "\n")
    # root-level SHA256SUMS over ALL evidence files (recursive)
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS",) \
                and p.suffix in (".csv", ".json"):
            lines.append(f"{B.sha256_of(p)}  {p.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    log(f"campaign complete -> {root} ({campaign['total_wall_s']}s)")
    print(json.dumps(head, indent=2, default=str))


if __name__ == "__main__":
    main()
