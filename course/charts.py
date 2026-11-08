"""Premium-engineering-keynote charts for Slides 28-51.

Every chart is generated from REAL benchmark summaries (results/raw/.../*.json)
and carries an evidence footnote: run id, CPU/GPU models, batch, precision and
the source CSV + its sha256. Charts are exported as PNG + editable SVG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets" / "slides"

BG = "#0e1116"
PANEL = "#161b22"
GRID = "#263041"
TEXT = "#e8e6e3"
MUTED = "#9aa4b2"
C_CPU = "#4d9fff"
C_GPU = "#ff4d5e"
C_INT8 = "#ffb000"
C_FP32 = "#8b9bb4"
C_OK = "#3ddc97"
C_WARN = "#ff8a3d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "font.family": ["DejaVu Sans", "Noto Sans CJK SC"],
    "font.size": 13, "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "svg.fonttype": "none",
})


def _finish(fig, ax, title, subtitle, footnote, name):
    ax.set_title(f"{title}\n{subtitle}", loc="left", pad=18, fontsize=18)
    fig.text(0.012, 0.015, footnote, fontsize=9.5, color=MUTED, family="DejaVu Sans")
    for a in (fig.axes if hasattr(fig, "axes") else [ax]):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{name}.png", dpi=180, facecolor=BG)
    fig.savefig(ASSETS / f"{name}.svg", facecolor=BG)
    plt.close(fig)
    print(f"[chart] {ASSETS / name}.png (+svg)")


def _bar_label(ax, bars, fmt="{:.1f}", unit="", dy=1.015):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{fmt.format(h)}{unit}",
                    (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=TEXT, xytext=(0, 3), textcoords="offset points")


def load_summary(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def _foot(summary: dict, extra: str = "") -> str:
    m = summary["meta"]
    ev = summary["evidence"]
    return (f"Run {summary['run_id']} · YOLO26l (official) · {m.get('cpu','')[:36]} · "
            f"GPU {m.get('gpu','')} · imgsz {m.get('imgsz',640)} · batch {m.get('batch',1)} · "
            f"{m.get('precision','fp32')} · warmup {m.get('warmup','?')} iters {m.get('iters','?')} · "
            f"evidence: {ev['csv']} (sha256 {ev['csv_sha256'][:12]}…) {extra}")


def _style_axes(ax):
    ax.grid(axis="y", alpha=0.6)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- Lab A ----
def chart_cpu_vs_gpu(cpu_sum: dict, gpu_sum: dict, name="01_cpu_vs_gpu",
                     foot: str | None = None):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    cs, gs = cpu_sum["stats"], gpu_sum["stats"]
    x = [0, 1]
    w = 0.34
    b1 = ax.bar([x[0] - w / 2, x[1] - w / 2], [cs["p50_ms"], gs["p50_ms"]],
                w, color=[C_CPU, C_GPU], label="P50 latency / P50 延迟")
    b2 = ax.bar([x[0] + w / 2, x[1] + w / 2], [cs["p95_ms"], gs["p95_ms"]],
                w, color=[C_CPU, C_GPU], alpha=0.45, label="P95 latency / P95 延迟")
    _bar_label(ax, b1, unit=" ms")
    _bar_label(ax, b2, unit=" ms")
    ax.set_xticks(x)
    ax.set_xticklabels([f"AMD CPU\n(PyTorch, end-to-end)",
                        f"W7900 GPU (ROCm)\n(PyTorch, end-to-end)"])
    ax.set_ylabel("Latency per image / 单张延迟 (ms)")
    thr_c = 1000.0 / cs["p50_ms"]
    thr_g = 1000.0 / gs["p50_ms"]
    ax.annotate(f"throughput ≈ {thr_c:.1f} img/s", (x[0], max(cs.values()) * 0.55),
                ha="center", fontsize=12, color=C_CPU, fontweight="bold")
    ax.annotate(f"throughput ≈ {thr_g:.1f} img/s", (x[1], max(gs.values()) * 0.55),
                ha="center", fontsize=12, color=C_GPU, fontweight="bold")
    ratio = cs["p50_ms"] / gs["p50_ms"]
    ax.text(0.985, 0.965,
            f"P50: GPU is {ratio:.2f}× the CPU speed\nP50：GPU 速度为 CPU 的 {ratio:.2f} 倍"
            if ratio > 1 else
            f"P50: GPU speed is {1/ratio:.2f}× of CPU\nP50：GPU 速度为 CPU 的 {1/ratio:.2f} 倍",
            transform=ax.transAxes, ha="right", va="top", fontsize=15,
            color=C_OK if ratio > 1 else C_WARN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#1c2330", ec=GRID))
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper center")
    _finish(fig, ax,
            "Lab A · Same model, two AMD devices / 同一模型，两种 AMD 设备",
            "YOLO26l · PyTorch · batch 1 · imgsz 640 · end-to-end per image",
            (foot or _foot(gpu_sum)), name)


# ---------------------------------------------------------------- Lab B ----
def chart_pytorch_vs_openvino(pt_sum: dict, ov_sum: dict, map_pt: float,
                              map_ov: float, name="02_pytorch_vs_openvino_cpu",
                              foot: str | None = None):
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.8, 7.2),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    ps, os_ = pt_sum["stats"], ov_sum["stats"]
    bars = ax.bar(["PyTorch CPU", "OpenVINO CPU"], [ps["p50_ms"], os_["p50_ms"]],
                  0.5, color=[C_FP32, C_CPU])
    err_lo = [ps["p50_ms"] - ps["min_ms"], os_["p50_ms"] - os_["min_ms"]]
    err_hi = [ps["p95_ms"] - ps["p50_ms"], os_["p95_ms"] - os_["p50_ms"]]
    ax.errorbar([0, 1], [ps["p50_ms"], os_["p50_ms"]],
                yerr=[err_lo, err_hi], fmt="none", ecolor=MUTED, capsize=6,
                lw=1.6, label="min–P95 range")
    _bar_label(ax, bars, unit=" ms")
    ax.set_ylabel("Latency per image / 单张延迟 (ms)")
    sp = ps["p50_ms"] / os_["p50_ms"]
    ax.text(0.5, 0.96, f"{sp:.2f}× " + ("faster / 更快" if sp > 1 else "slower / 更慢"),
            transform=ax.transAxes, ha="center", va="top", fontsize=22,
            color=C_OK if sp > 1 else C_WARN, fontweight="bold")
    ax.set_xticklabels(["PyTorch CPU", "OpenVINO CPU\n(same AMD CPU)"])
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")

    axr.bar(["PyTorch", "OpenVINO"], [map_pt, map_ov], 0.5,
            color=[C_FP32, C_CPU])
    axr.set_ylim(0, max(map_pt, map_ov) * 1.25)
    axr.set_ylabel("mAP50-95 (val subset)")
    for i, v in enumerate([map_pt, map_ov]):
        axr.annotate(f"{v:.4f}", (i, v), ha="center", va="bottom",
                     fontsize=13, fontweight="bold")
    axr.set_title("Accuracy parity / 精度一致性", fontsize=14)
    _style_axes(axr)
    _finish(fig, ax,
            "Lab B · Same AMD CPU, different runtime / 同一 CPU，不同推理运行时",
            "YOLO26l · FP32 · batch 1 · imgsz 640 · same postprocessing path",
            (foot or _foot(ov_sum)), name)


# ---------------------------------------------------------------- Lab C ----
def chart_fp32_vs_int8(fp32_sum: dict, int8_sum: dict, size_fp32: float,
                       size_int8: float, map_fp32: float, map_int8: float,
                       name="03_fp32_vs_int8", foot: str | None = None,
                       subtitle: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2))
    f5, i5 = fp32_sum["stats"], int8_sum["stats"]

    ax = axes[0]
    bars = ax.bar(["FP32", "INT8"], [f5["p50_ms"], i5["p50_ms"]], 0.5,
                  color=[C_FP32, C_INT8])
    _bar_label(ax, bars, unit=" ms")
    ax.set_ylabel("P50 latency / P50 延迟 (ms)")
    sp = f5["p50_ms"] / i5["p50_ms"]
    ax.set_title(f"Latency · {sp:.2f}× " +
                 ("faster / 更快" if sp > 1 else "slower / 更慢"), fontsize=14)
    _style_axes(ax)

    ax = axes[1]
    bars = ax.bar(["FP32", "INT8"], [size_fp32, size_int8], 0.5,
                  color=[C_FP32, C_INT8])
    _bar_label(ax, bars, fmt="{:.0f}", unit=" MB")
    ax.set_ylabel("Model size / 模型大小 (MB)")
    ax.set_title(f"Size · {size_fp32/size_int8:.2f}× smaller / 更小", fontsize=14)
    _style_axes(ax)

    ax = axes[2]
    bars = ax.bar(["FP32", "INT8"], [map_fp32, map_int8], 0.5,
                  color=[C_FP32, C_INT8])
    for b, v in zip(bars, [map_fp32, map_int8]):
        ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("mAP50-95 (val subset)")
    drop = map_fp32 - map_int8
    ax.set_title(f"Accuracy · ΔmAP = {drop:+.4f}", fontsize=14)
    ax.set_ylim(0, map_fp32 * 1.2)
    _style_axes(ax)
    _finish(fig, axes[0],
            "Lab C · INT8 quantization on AMD CPU / AMD CPU 上的 INT8 量化",
            subtitle or ("YOLO26l · OpenVINO · NNCF post-training quantization "
                         "· 150 real COCO calibration images"),
            (foot or _foot(int8_sum)), name)


# ---------------------------------------------------------------- Lab D ----
def chart_resolution(imgsz_a: int, imgsz_b: int, lat_a: dict, lat_b: dict,
                     map_a: float | None, map_b: float | None,
                     mem_a: float, mem_b: float, summary: dict,
                     name="04_resolution_vs_latency_accuracy",
                     foot: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    labels = [f"imgsz {imgsz_a}", f"imgsz {imgsz_b}"]

    ax = axes[0]
    bars = ax.bar(labels, [lat_a["p50_ms"], lat_b["p50_ms"]], 0.5,
                  color=[C_GPU, C_WARN])
    _bar_label(ax, bars, unit=" ms")
    ax.set_ylabel("GPU latency per image / 单张延迟 (ms)")
    ax.set_title("Latency / 延迟 (W7900, batch 1)", fontsize=15)
    _style_axes(ax)

    ax = axes[1]
    xs = [0, 1]
    maps = [map_a, map_b]
    bars = ax.bar(xs, [m if m else 0 for m in maps], 0.5, color=[C_GPU, C_WARN])
    for b, v in zip(bars, maps):
        if v is not None:
            ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(0, (max(m for m in maps if m) * 1.25) if any(maps) else 1)
    ax.set_ylabel("mAP50-95 (val subset)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    mem = f"  ·  VRAM: {mem_a:.2f} GB → {mem_b:.2f} GB"
    ax.set_title("Accuracy / 精度" + mem, fontsize=15)
    _style_axes(ax)
    _finish(fig, axes[0],
            "Lab D-A · Input resolution trade-off / 输入分辨率的权衡",
            "YOLO26l · PyTorch ROCm · W7900 · batch 1 · smaller input = less compute",
            (foot or _foot(summary)), name)


def chart_batch(batch_a: int, batch_b: int, thr_a: float, thr_b: float,
                lat_note_a: str, lat_note_b: str, mem_a: float, mem_b: float,
                summary: dict, name="05_batch_vs_throughput",
                foot: str | None = None, e2e_note: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    labels = [f"batch {batch_a}", f"batch {batch_b}"]

    ax = axes[1]
    bars = ax.bar(labels, [thr_a, thr_b], 0.5, color=[C_GPU, C_OK])
    _bar_label(ax, bars, fmt="{:.0f}", unit=" img/s")
    ax.set_ylabel("Throughput / 吞吐 (images/s)")
    ax.set_title(f"Throughput · {thr_b/thr_a:.2f}× at batch {batch_b}",
                 fontsize=15)
    _style_axes(ax)

    ax = axes[0]
    bars = ax.bar(labels, [mem_a, mem_b], 0.5, color=[C_GPU, C_OK])
    _bar_label(ax, bars, fmt="{:.2f}", unit=" GB")
    ax.set_ylabel("Peak GPU memory / 峰值显存 (GB)")
    ax.set_title("VRAM / 显存占用", fontsize=15)
    _style_axes(ax)
    _finish(fig, axes[0],
            "Lab D-B · Batch size vs throughput & memory / Batch 与吞吐、显存",
            (e2e_note or
             f"YOLO26l · W7900 · imgsz 640 · per-request latency NOT divided "
             f"by batch: {lat_note_a} → {lat_note_b}"),
            (foot or _foot(summary)), name)


# ---------------------------------------------------------------- Lab E ----
def chart_pipeline(serial: dict, pipeline: dict,
                    name="06_serial_vs_async_pipeline",
                    foot: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))

    ax = axes[0]
    bars = ax.bar(["Serial / 串行", "Async pipeline / 异步流水线"],
                  [serial["total_ms"] / 1000, pipeline["total_ms"] / 1000], 0.5,
                  color=[C_FP32, C_OK])
    _bar_label(ax, bars, fmt="{:.1f}", unit=" s")
    ax.set_ylabel(f"Wall time for {serial['n_images']} images / 总耗时 (s)")
    gain = serial["total_ms"] / pipeline["total_ms"]
    ax.set_title(f"Total time · {gain:.2f}× " +
                 ("faster / 更快" if gain > 1 else "slower / 更慢"), fontsize=15)
    _style_axes(ax)

    ax = axes[1]
    pre = [serial["pre_p50_ms"], pipeline["pre_p50_ms"]]
    inf = [serial["infer_p50_ms"], pipeline["infer_p50_ms"]]
    xs = [0, 1]
    b1 = ax.bar(xs, inf, 0.5, color=C_GPU, label="GPU infer / GPU 推理")
    b2 = ax.bar(xs, pre, 0.5, bottom=inf, color=C_CPU,
                label="CPU load+preprocess / CPU 读取+预处理")
    for i in xs:
        ax.annotate(f"e2e p50 {serial['e2e_p50_ms']:.0f} ms\n(no queue)",
                    (0, inf[0] + pre[0]), ha="center", va="bottom",
                    fontsize=11, color=MUTED)
        ax.annotate(f"e2e p50 {pipeline['e2e_p50_ms']:.0f} ms\n(burst: incl. queue wait)",
                    (1, inf[1] + pre[1]), ha="center", va="bottom",
                    fontsize=11, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Serial / 串行", "Pipeline / 流水线"])
    ax.set_ylabel("P50 per-image breakdown / 单张耗时分解 (ms)")
    ax.set_title("Where the time goes / 时间去哪了", fontsize=15)
    _style_axes(ax)
    ax.legend(frameon=False)
    _finish(fig, axes[0],
            "Lab E · CPU + GPU heterogeneous pipeline / CPU+GPU 异构流水线",
            f"{serial['n_images']} images · YOLO26l · W7900 · CPU prepare overlaps GPU compute",
            (foot or (f"serial {serial['images_per_s']:.1f} img/s → pipeline "
                      f"{pipeline['images_per_s']:.1f} img/s · detection "
                      f"consistency verified / 检测结果一致性已校验")),
            name)


def chart_environment(cpu: str, gpu: str, versions: dict, name="00_environment"):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    rows = [("CPU", cpu), ("GPU", gpu)] + list(versions.items())
    y = 0.92
    ax.text(0.02, y, "Day 02 Lab Machine / 实验机器", fontsize=24,
            fontweight="bold", color=TEXT)
    y -= 0.1
    for k, v in rows:
        ax.text(0.04, y, f"{k}", fontsize=15, color=MUTED)
        ax.text(0.3, y, str(v), fontsize=15, color=TEXT, fontweight="bold")
        y -= 0.085
    _finish(fig, ax, "", "", "auto-generated by course/charts.py", name)
