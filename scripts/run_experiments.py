#!/usr/bin/env python3
"""Headless benchmark runner behind scripts/run_all.sh.

Profiles:
  quick -- classroom mode: fewer iterations/subsets, same official YOLO26l.
  full  -- formal, citable benchmark: full repetitions and full subsets.

Both profiles use the SAME weights, images and code paths; only repetition
counts / subset sizes differ. All numbers come from real runs on this machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from course import benchmark as B          # noqa: E402
from course import charts, model as M      # noqa: E402
from course import pipeline as P           # noqa: E402
from course import quantization as Q       # noqa: E402
from course import setup_checks as SC      # noqa: E402

RESULTS_LATEST = REPO / "results" / "latest.json"


def log(msg: str) -> None:
    print(f"\033[1;36m[run {datetime.now().strftime('%H:%M:%S')}]\033[0m {msg}",
          flush=True)


def cycle_images(n: int):
    imgs = B.load_bench_images()
    while True:
        yield from imgs


def exp_environment(out: dict) -> None:
    log("Environment report")
    rep = SC.full_report()
    SC.assert_healthy(rep)
    out["environment"] = rep
    charts.chart_environment(
        cpu=rep["cpu"]["model"], gpu=rep.get("device_name", "gfx1100"),
        versions={"ROCm/HIP": rep.get("hip"), "PyTorch": rep.get("torch"),
                  "OpenVINO": rep.get("openvino"), "Ultralytics": rep.get("ultralytics"),
                  "Python": rep.get("python")})


def exp_lab_a(out: dict, warmup: int, iters: int) -> None:
    log(f"Lab A · CPU vs GPU baseline (warmup={warmup}, iters={iters})")
    from ultralytics import YOLO
    imgs = B.load_bench_images()
    it = cycle_images(iters)
    nthreads = B.pin_torch_threads(8)
    log(f"Lab A · PyTorch CPU end-to-end (torch threads pinned to {nthreads}, "
        f"cgroup quota {B.cgroup_cpu_quota()} cores)")

    model = YOLO(str(M.OFFICIAL_PT))
    gen = iter(list(next(it) for _ in range(iters)))
    cpu_samples = []
    for i in range(iters):
        p = imgs[i % len(imgs)]
        if i < warmup:
            model.predict(source=str(p), imgsz=640, device="cpu",
                          verbose=False)
    for i in range(iters):
        p = imgs[i % len(imgs)]
        cpu_samples.append(B.timed_call(
            lambda p=p: model.predict(source=str(p), imgsz=640, device="cpu",
                                      verbose=False), sync=False))
    cpu_sum = B.save_run(B.RunMeta(
        experiment="labA_cpu_vs_gpu", variant="pytorch_cpu", device="cpu",
        runtime="pytorch-rocm-cpu", warmup=warmup, iters=iters,
        notes="end-to-end per image (pre+infer+post)"), cpu_samples)

    log("Lab A · PyTorch ROCm GPU end-to-end")
    gpu_samples = []
    for i in range(warmup + iters):
        p = imgs[i % len(imgs)]
        t = B.timed_call(
            lambda p=p: model.predict(source=str(p), imgsz=640, device="cuda",
                                      verbose=False), sync=True)
        if i >= warmup:
            gpu_samples.append(t)
    gpu_sum = B.save_run(B.RunMeta(
        experiment="labA_cpu_vs_gpu", variant="pytorch_rocm_gpu", device="cuda(gfx1100)",
        runtime="pytorch-rocm", warmup=warmup, iters=iters,
        notes="end-to-end per image (pre+infer+post), cuda-synced"), gpu_samples)

    out["labA"] = {"cpu": cpu_sum, "gpu": gpu_sum}
    charts.chart_cpu_vs_gpu(cpu_sum, gpu_sum)
    cs, gs = cpu_sum["stats"], gpu_sum["stats"]
    log(f"  CPU P50 {cs['p50_ms']:.1f} ms · GPU P50 {gs['p50_ms']:.1f} ms "
        f"· ratio {cs['p50_ms']/gs['p50_ms']:.2f}x")


def exp_lab_b(out: dict, warmup: int, iters: int, map_images: str) -> None:
    log(f"Lab B · PyTorch vs OpenVINO on the same AMD CPU")
    from ultralytics import YOLO
    imgs = B.load_bench_images()

    ov_dir = M.export_openvino_fp32(640)
    t0 = time.perf_counter()
    pt_model = YOLO(str(M.OFFICIAL_PT))
    t_load_pt = time.perf_counter() - t0

    cpu_samples = []
    for i in range(warmup):
        model = pt_model
        p = imgs[i % len(imgs)]
        model.predict(source=str(p), imgsz=640, device="cpu", verbose=False)
    for i in range(iters):
        p = imgs[i % len(imgs)]
        cpu_samples.append(B.timed_call(
            lambda p=p: pt_model.predict(source=str(p), imgsz=640, device="cpu",
                                         verbose=False), sync=False))
    pt_sum = B.save_run(B.RunMeta(
        experiment="labB_runtime_cpu", variant="pytorch_cpu", device="cpu",
        runtime="pytorch-rocm-cpu", warmup=warmup, iters=iters,
        extra={"model_load_s": round(t_load_pt, 3)}), cpu_samples)

    t0 = time.perf_counter()
    ov_model = YOLO(str(ov_dir))
    ov_model.predict(source=str(imgs[0]), imgsz=640, device="cpu",
                     verbose=False)
    t_compile = time.perf_counter() - t0
    ov_samples = []
    for i in range(warmup + iters):
        p = imgs[i % len(imgs)]
        t = B.timed_call(
            lambda p=p: ov_model.predict(source=str(p), imgsz=640, device="cpu",
                                         verbose=False), sync=False)
        if i >= warmup:
            ov_samples.append(t)
    ov_sum = B.save_run(B.RunMeta(
        experiment="labB_runtime_cpu", variant="openvino_cpu", device="cpu",
        runtime="openvino-cpu", warmup=warmup, iters=iters,
        notes="same CPU, same weights exported from same yolo26l.pt",
        extra={"first_load_and_compile_s": round(t_compile, 3)}), ov_samples)

    import torch
    threads = torch.get_num_threads()
    log(f"Lab B · mAP on {map_images} val subset (torch threads={threads})")
    map_pt = B.map5095(pt_model, str(REPO / "data/val_subset.yaml"), 640, 8, "cpu")
    map_ov = B.map5095(ov_model, str(REPO / "data/val_subset.yaml"), 640, 8, "cpu")
    out["labB"] = {"pytorch_cpu": pt_sum, "openvino_cpu": ov_sum,
                   "map": {"pytorch": map_pt, "openvino": map_ov},
                   "cpu_threads": threads}
    charts.chart_pytorch_vs_openvino(
        pt_sum, ov_sum, map_pt["map50_95"], map_ov["map50_95"])
    ps, os_ = pt_sum["stats"], ov_sum["stats"]
    log(f"  PyTorch P50 {ps['p50_ms']:.1f} ms · OpenVINO P50 {os_['p50_ms']:.1f} ms "
        f"· ratio {ps['p50_ms']/os_['p50_ms']:.2f}x · mAP {map_pt['map50_95']:.4f} "
        f"vs {map_ov['map50_95']:.4f}")


def exp_lab_c(out: dict, warmup: int, iters: int, calib_images: int,
              map_device: str) -> None:
    log(f"Lab C · OpenVINO FP32 vs INT8 (calib={calib_images}, preset=MIXED)")
    from ultralytics import YOLO
    imgs = B.load_bench_images()
    fp32_dir = M.export_openvino_fp32(640)
    t0 = time.perf_counter()
    int8_dir = Q.quantize_int8(n_calib_images=calib_images,
                               subset_size=calib_images)
    t_quantize = time.perf_counter() - t0

    fp32_model = YOLO(str(fp32_dir))
    int8_model = YOLO(str(int8_dir))

    def bench_model(m, variant, extra=None):
        for i in range(warmup):
            p = imgs[i % len(imgs)]
            m.predict(source=str(p), imgsz=640, device="cpu", verbose=False)
        samples = []
        for i in range(iters):
            p = imgs[i % len(imgs)]
            samples.append(B.timed_call(
                lambda p=p: m.predict(source=str(p), imgsz=640, device="cpu",
                                      verbose=False), sync=False))
        return B.save_run(B.RunMeta(
            experiment="labC_int8", variant=variant, device="cpu",
            runtime="openvino-cpu", precision=variant.split("_")[-1],
            warmup=warmup, iters=iters, extra=extra or {}), samples)

    fp32_sum = bench_model(fp32_model, "openvino_fp32")
    int8_sum = bench_model(int8_model, "openvino_int8",
                           {"quantization_time_s": round(t_quantize, 1),
                            "calibration_images": calib_images,
                            "nncf_preset": "mixed"})

    size_fp32 = Q.model_dir_size_mb(fp32_dir)
    size_int8 = Q.model_dir_size_mb(int8_dir)
    log("Lab C · mAP (OpenVINO CPU validation)")
    map_fp32 = B.map5095(fp32_model, str(REPO / "data/val_subset.yaml"), 640, 8, "cpu")
    map_int8 = B.map5095(int8_model, str(REPO / "data/val_subset.yaml"), 640, 8, "cpu")
    out["labC"] = {"fp32": fp32_sum, "int8": int8_sum,
                   "size_mb": {"fp32": size_fp32, "int8": size_int8},
                   "map": {"fp32": map_fp32, "int8": map_int8},
                   "quantization_time_s": round(t_quantize, 1)}
    charts.chart_fp32_vs_int8(fp32_sum, int8_sum, size_fp32, size_int8,
                              map_fp32["map50_95"], map_int8["map50_95"])
    f5, i5 = fp32_sum["stats"], int8_sum["stats"]
    log(f"  FP32 P50 {f5['p50_ms']:.1f} ms · INT8 P50 {i5['p50_ms']:.1f} ms · "
        f"ratio {f5['p50_ms']/i5['p50_ms']:.2f}x · size {size_fp32:.0f}->{size_int8:.0f} MB "
        f"· mAP {map_fp32['map50_95']:.4f}->{map_int8['map50_95']:.4f}")


def exp_lab_d(out: dict, warmup: int, iters: int, map_device: str) -> None:
    log("Lab D · GPU resolution (640 vs 512) and batch (1 vs 8)")
    from ultralytics import YOLO
    imgs = B.load_bench_images()
    model = YOLO(str(M.OFFICIAL_PT))

    def gpu_bench(imgsz: int):
        for i in range(warmup):
            model.predict(source=str(imgs[i % len(imgs)]), imgsz=imgsz,
                          device="cuda", verbose=False)
        import torch
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for i in range(iters):
            p = imgs[i % len(imgs)]
            samples.append(B.timed_call(
                lambda p=p: model.predict(source=str(p), imgsz=imgsz,
                                          device="cuda", verbose=False), sync=True))
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        return samples, mem

    s640, mem640 = gpu_bench(640)
    sum640 = B.save_run(B.RunMeta(
        experiment="labD_resolution", variant="imgsz640", device="cuda(gfx1100)",
        runtime="pytorch-rocm", imgsz=640, warmup=warmup, iters=iters), s640)
    s512, mem512 = gpu_bench(512)
    sum512 = B.save_run(B.RunMeta(
        experiment="labD_resolution", variant="imgsz512", device="cuda(gfx1100)",
        runtime="pytorch-rocm", imgsz=512, warmup=warmup, iters=iters), s512)

    yaml = str(REPO / "data/val_subset.yaml")
    map640 = B.map5095(model, yaml, 640, 8, "cuda")
    map512 = B.map5095(model, yaml, 512, 8, "cuda")

    # ---- Experiment B: batch 1 vs 8 (forward-time on W7900, honest metrics) --
    import torch
    stages = P.Yolo26lStages(imgsz=640)

    def batch_forward(paths, bs):
        im0s = [stages.load(p) for p in paths]
        im = stages.preprocess(im0s)
        torch.cuda.synchronize()
        t0 = B.now_ms()
        preds = stages.infer(im)
        torch.cuda.synchronize()
        return B.now_ms() - t0, preds, im, im0s

    window = imgs[:8]
    for _ in range(warmup):
        batch_forward(window, 8)
    torch.cuda.reset_peak_memory_stats()
    b1_samples = []
    for i in range(iters):
        t, _, _, _ = batch_forward([imgs[(i) % len(imgs)]], 1)
        b1_samples.append(t)
    b8_samples = []
    for i in range(iters):
        t, _, _, _ = batch_forward(window, 8)
        b8_samples.append(t)
    mem_b = torch.cuda.max_memory_allocated() / 1024 ** 3

    thr1 = 1000.0 / B.pct(b1_samples, 50)
    thr8 = 8 * 1000.0 / B.pct(b8_samples, 50)
    sumb1 = B.save_run(B.RunMeta(
        experiment="labD_batch", variant="batch1_forward", device="cuda(gfx1100)",
        runtime="pytorch-rocm", batch=1, warmup=warmup, iters=iters,
        notes="GPU forward only, batch=1"), b1_samples)
    sumb8 = B.save_run(B.RunMeta(
        experiment="labD_batch", variant="batch8_forward", device="cuda(gfx1100)",
        runtime="pytorch-rocm", batch=8, warmup=warmup, iters=iters,
        notes="GPU forward only, whole batch of 8 timed as one call"), b8_samples)

    out["labD"] = {
        "resolution": {
            "640": {"stats": sum640["stats"], "map": map640, "vram_gb": mem640},
            "512": {"stats": sum512["stats"], "map": map512, "vram_gb": mem512},
            "summaries": [sum640["run_id"], sum512["run_id"]]},
        "batch": {
            "1": {"p50_ms": B.pct(b1_samples, 50), "throughput_img_s": thr1},
            "8": {"p50_batch_ms": B.pct(b8_samples, 50),
                  "per_image_equivalent_ms": B.pct(b8_samples, 50) / 8,
                  "throughput_img_s": thr8,
                  "note": "batch completion time / 8 is an equivalent cost, "
                          "NOT a single-request latency"},
            "vram_gb": mem_b,
            "summaries": [sumb1["run_id"], sumb8["run_id"]]},
    }
    charts.chart_resolution(640, 512, sum640["stats"], sum512["stats"],
                            map640["map50_95"], map512["map50_95"],
                            mem640, mem512, sum512)
    charts.chart_batch(
        1, 8, thr1, thr8,
        lat_note_a=f"{B.pct(b1_samples,50):.0f} ms/req",
        lat_note_b=f"{B.pct(b8_samples,50):.0f} ms/batch",
        mem_a=min(mem640, mem_b), mem_b=mem_b, summary=sumb8)
    log(f"  640 P50 {sum640['stats']['p50_ms']:.1f} ms · 512 P50 "
        f"{sum512['stats']['p50_ms']:.1f} ms · mAP {map640['map50_95']:.4f} "
        f"-> {map512['map50_95']:.4f}")
    log(f"  batch1 {thr1:.1f} img/s · batch8 {thr8:.1f} img/s "
        f"({thr8/thr1:.2f}x) · VRAM {mem_b:.2f} GB")


def exp_lab_e(out: dict, n_images: int) -> None:
    log(f"Lab E · serial vs async pipeline on {n_images} images")
    imgs = B.load_bench_images(n_images)
    serial = P.run_serial(imgs, imgsz=640)
    pipeline = P.run_pipeline(imgs, imgsz=640)
    cmp = P.compare_results(serial, pipeline)
    out["labE"] = {"serial": serial, "pipeline": pipeline, "compare": cmp}
    charts.chart_pipeline(serial, pipeline)
    log(f"  serial {serial['total_ms']/1000:.1f} s ({serial['images_per_s']:.1f} img/s) · "
        f"pipeline {pipeline['total_ms']/1000:.1f} s ({pipeline['images_per_s']:.1f} img/s) · "
        f"gain {cmp['throughput_gain_x']:.2f}x · consistent={cmp['consistent']}")
    if not cmp["consistent"]:
        log(f"  WARN detection mismatches: {list(cmp['detection_mismatches'].items())[:5]}")


EXPERIMENTS = {
    "environment": lambda out, cfg: exp_environment(out),
    "labA": lambda out, cfg: exp_lab_a(out, cfg["warmup"], cfg["iters"]),
    "labB": lambda out, cfg: exp_lab_b(out, cfg["warmup"], cfg["iters"], cfg["map_images"]),
    "labC": lambda out, cfg: exp_lab_c(out, cfg["warmup"], cfg["iters"],
                                       cfg["calib_images"], cfg["map_device"]),
    "labD": lambda out, cfg: exp_lab_d(out, cfg["warmup"], cfg["iters"], cfg["map_device"]),
    "labE": lambda out, cfg: exp_lab_e(out, cfg["pipeline_images"]),
}

PROFILES = {
    "quick": {"warmup": 3, "iters": 10, "map_images": 60,
              "calib_images": 100, "map_device": "cuda", "pipeline_images": 60,
              },
    "full": {"warmup": 10, "iters": 50, "map_images": 120,
             "calib_images": 150, "map_device": "cuda", "pipeline_images": 100,
             },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="carry forward previous same-profile results "
                         "(default: fresh latest.json every invocation)")
    args = ap.parse_args()
    cfg = PROFILES[args.profile]

    # v2.0.1 fix (defect B-07): v2.0.0 silently merged results from previous
    # runs, mixing run ids across invocations. Now a fresh start is the
    # default; --resume is the explicit opt-in for continuing an interrupted
    # FULL run of the same profile.
    out = {"profile": args.profile,
           "generated": datetime.now().isoformat(timespec="seconds")}
    if args.resume and RESULTS_LATEST.exists():
        prev = json.loads(RESULTS_LATEST.read_text())
        if prev.get("profile") == args.profile:
            out.update({k: v for k, v in prev.items()
                        if k not in ("generated", "profile")})
            out["resumed_from"] = prev.get("generated")
    out["profile"] = args.profile
    out["generated"] = datetime.now().isoformat(timespec="seconds")

    t0 = time.perf_counter()
    for name in args.only.split(","):
        name = name.strip()
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name}")
        EXPERIMENTS[name](out, cfg)
        RESULTS_LATEST.write_text(json.dumps(out, indent=2, default=str))
    log(f"all done in {time.perf_counter()-t0:.0f}s -> results/latest.json "
        f"+ assets/slides/*.png")


if __name__ == "__main__":
    main()
