"""Shared, scientifically honest benchmark harness for all six labs.

Rules encoded here (see docs/BENCHMARK_PROTOCOL.md):
  * warm-up runs are excluded from statistics and their count is recorded;
  * GPU timing always synchronizes before/after (ROCm is asynchronous);
  * raw per-iteration samples are saved to CSV -- no dropping of slow samples;
  * every run writes a summary.json with full environment + parameter metadata;
  * first-load time (model load / compile) is measured separately, never
    mixed into inference latency.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "raw"


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def pct(samples: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0,100]."""
    s = sorted(samples)
    if not s:
        raise ValueError("no samples")
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def stats_of(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": pct(samples_ms, 50),
        "p95_ms": pct(samples_ms, 95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


# --------------------------------------------------------------------------
# timing primitives
# --------------------------------------------------------------------------
def now_ms() -> float:
    return time.perf_counter() * 1000.0


def cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed_call(fn: Callable[[], object], sync: bool = True) -> float:
    """Time one call in ms; synchronizes GPU first/last when sync=True."""
    if sync:
        cuda_sync()
    t0 = now_ms()
    fn()
    if sync:
        cuda_sync()
    return now_ms() - t0


def bench(fn: Callable[[], object], warmup: int, iters: int,
          sync: bool = True) -> list[float]:
    """Run warmup (untimed) then iters timed calls; returns raw ms samples."""
    for _ in range(warmup):
        fn()
    if sync:
        cuda_sync()
    samples = [timed_call(fn, sync=sync) for _ in range(iters)]
    return samples


# --------------------------------------------------------------------------
# run records
# --------------------------------------------------------------------------
@dataclass
class RunMeta:
    experiment: str
    variant: str
    device: str                    # "cpu" | "cuda(gfx1100)"
    runtime: str                   # "pytorch-rocm" | "openvino-cpu" | ...
    model: str = "yolo26l.pt (official, 80-class COCO det)"
    model_sha256: str = ""
    imgsz: int = 640
    batch: int = 1
    precision: str = "fp32"
    warmup: int = 5
    iters: int = 30
    num_images: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = datetime.now().isoformat(timespec="seconds")
        d["cpu"] = _cpu_model()
        d["gpu"] = _gpu_name()
        try:
            from . import setup_checks
            d["env"] = {k: setup_checks.full_report().get(k)
                        for k in ("torch", "hip", "openvino", "ultralytics")}
        except Exception:
            pass
        if not d["model_sha256"]:
            from . import model as M
            d["model_sha256"] = M.model_sha256()
        return d


def _cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":")[1].strip()
    except Exception:
        pass
    return "unknown"


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "none"


def rss_mb() -> float:
    """Peak resident set size of this process tree, in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0


def gpu_mem_peak_gb(reset: bool = False) -> float:
    """Peak GPU memory allocated by torch on this device, in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            if reset:
                torch.cuda.reset_peak_memory_stats()
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    return 0.0


def save_run(meta: RunMeta, samples_ms: list[float],
             derived: dict | None = None) -> dict:
    """Persist a benchmark run: per-iteration CSV + summary JSON.

    Evidence chain (v2.0.1, fixes the v2.0.0 self-hash bug):
      * the CSV is the immutable measurement record;
      * the summary JSON *references* the CSV sha256 but never its own hash
        (a file cannot contain its own hash without circular rewrites);
      * after the JSON is finalized, its sha256 is recorded in the
        experiment-directory SHA256SUMS file (together with the CSV's);
      * `scripts/verify_evidence.py` recomputes every hash from bytes.

    Returns the summary dict (evidence.json_sha256, when present, is filled
    by refresh_sums() and matches the file as finally written).
    """
    run_id = f"{meta.experiment}_{meta.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS / meta.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_id}.csv"
    json_path = out_dir / f"{run_id}_summary.json"

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    lines = ["iteration,ms"] + [f"{i+1},{ms:.4f}"
                                for i, ms in enumerate(samples_ms)]
    csv_path.write_text("\n".join(lines) + "\n")

    summary = {
        "run_id": run_id,
        "meta": meta.to_dict(),
        "stats": stats_of(samples_ms),
        "derived": derived or {},
        "evidence": {
            "csv": _rel(csv_path),
            "csv_sha256": sha256_of(csv_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["evidence"]["json"] = _rel(json_path)
    summary["evidence"]["json_sha256"] = sha256_of(json_path)
    refresh_sums(out_dir)
    print(f"[saved] {_rel(csv_path)}  ({len(samples_ms)} samples)")
    return summary


def refresh_sums(out_dir: Path) -> Path:
    """Rewrite <out_dir>/SHA256SUMS covering every *.csv and *_summary.json.

    The sums file is regenerated from file bytes each time, so it is a
    tamper-evident snapshot of the whole experiment directory (verified by
    scripts/verify_evidence.py --campaign ...).
    """
    entries = []
    for p in sorted(out_dir.iterdir()):
        if p.suffix == ".csv" or p.name.endswith("_summary.json") \
                or "records" in p.name and p.suffix == ".json":
            entries.append(f"{sha256_of(p)}  {p.name}")
    sums = out_dir / "SHA256SUMS"
    sums.write_text("\n".join(entries) + "\n")
    return sums


# --------------------------------------------------------------------------
# campaign / environment controls (v2.0.1)
# --------------------------------------------------------------------------
def set_results_root(path: Path) -> None:
    """Redirect raw-run output (e.g. to validation/v2.0.1/campaigns/<id>/)."""
    global RESULTS
    RESULTS = Path(path)


def results_root() -> Path:
    return RESULTS


def host_load() -> dict:
    """Snapshot of host load + cgroup throttling counters (shared-machine
    disclosure for benchmarks; not exclusive hardware)."""
    out: dict = {}
    try:
        parts = Path("/proc/loadavg").read_text().split()
        out["loadavg_1_5_15"] = [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        pass
    try:
        d = dict(l.split() for l in Path("/sys/fs/cgroup/cpu.stat").read_text()
                 .splitlines())
        out["cgroup_nr_throttled"] = int(d.get("nr_throttled", -1))
        out["cgroup_throttled_usec"] = int(d.get("throttled_usec", -1))
    except Exception:
        pass
    return out


_QUIET_SCAN_LAST: dict = {}


def pick_quiet_cpus(n: int = 16, sample_s: float = 2.0) -> list[int]:
    """Pick n CPUs (same NUMA node when possible) with the lowest utilization.

    This machine's host is SHARED with other tenants; CPU benchmarks pin the
    process to a quiet core set and record it, to reduce neighbor contention.
    Selection is deterministic given the same /proc/stat snapshot.
    """
    def snap() -> dict[int, tuple[int, int]]:
        d = {}
        for ln in Path("/proc/stat").read_text().splitlines():
            if ln.startswith("cpu") and ln[3].isdigit():
                f = ln.split()
                vals = list(map(int, f[1:]))
                d[int(f[0][3:])] = (vals[3] + vals[4], sum(vals))
        return d
    a = snap()
    time.sleep(sample_s)
    b = snap()
    busy = {c: 1.0 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1])
            for c in a if c in b}
    allowed = sorted(os.sched_getaffinity(0))
    numa = _numa_cpu_map()
    best: list[int] = []
    for node_cpus in numa.values():
        cand = [c for c in node_cpus if c in busy]
        cand.sort(key=lambda c: busy[c])
        if len(cand) >= n:
            best = cand[:n]
            break
    if not best:
        best = sorted(allowed, key=lambda c: busy.get(c, 1.0))[:n]
    _QUIET_SCAN_LAST["picked"] = best
    _QUIET_SCAN_LAST["busy_of_picked"] = [round(busy.get(c, -1), 3) for c in best]
    return best


def pin_affinity(cpus: list[int] | None) -> list[int]:
    """Set (or restore empty=full-mask) CPU affinity; returns previous mask."""
    prev = sorted(os.sched_getaffinity(0))
    if cpus is None:
        os.sched_setaffinity(0, sorted(_NUMA_ALL_CPUS or prev))
    else:
        os.sched_setaffinity(0, sorted(cpus))
    return prev


def _numa_cpu_map() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    try:
        nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
        for nd in nodes:
            cl = nd / "cpulist"
            if cl.exists():
                out[nd.name] = _parse_cpulist(cl.read_text().strip())
    except Exception:
        out["all"] = sorted(os.sched_getaffinity(0))
    return out


def _parse_cpulist(s: str) -> list[int]:
    cpus: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


_NUMA_ALL_CPUS = sorted(os.sched_getaffinity(0))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_torch_threads(n: int = 8) -> int:
    """Pin PyTorch CPU threads for fair, reproducible CPU benchmarks.

    Ultralytics' select_device('cpu') resets torch threads to
    min(8, cpu_count-1) on every CPU predict; under this machine's container
    CPU quota (16 cores) we measured PyTorch fastest at 8 threads
    (8 -> 190 ms, 16 -> 193 ms, 32 -> 297 ms, 64 -> 516 ms per image).
    Setting ultralytics' NUM_THREADS keeps the value stable across calls.
    """
    import torch
    import ultralytics.utils as U
    U.NUM_THREADS = n
    torch.set_num_threads(n)
    return torch.get_num_threads()


def cgroup_cpu_quota() -> int | None:
    """Container CPU quota (cores), if any -- affects thread tuning."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return int(int(q) / int(p)) if q != "max" else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# ultralytics-level helpers shared by notebooks
# --------------------------------------------------------------------------
def bench_ultralytics_predict(model, img_path: Path, imgsz: int, device: str,
                              warmup: int, iters: int) -> list[float]:
    """End-to-end per-image latency (pre + infer + post) for a YOLO model."""
    def call() -> None:
        model.predict(source=str(img_path), imgsz=imgsz, device=device,
                      verbose=False)
    return bench(call, warmup, iters, sync=(device.startswith("cuda")))


def bench_ultralytics_batch(model, img_paths: list[Path], imgsz: int,
                            device: str, warmup: int, iters: int) -> list[float]:
    """Time full batches; each sample = one whole batch of len(img_paths)."""
    src = [str(p) for p in img_paths]

    def call() -> None:
        model.predict(source=src, imgsz=imgsz, device=device, verbose=False)
    return bench(call, warmup, iters, sync=(device.startswith("cuda")))


def load_bench_images(n: int | None = None) -> list[Path]:
    """Deterministic benchmark image list (the fixed COCO val subset)."""
    img_dir = REPO / "data" / "val_subset" / "images"
    paths = sorted(img_dir.glob("*.jpg"))
    if n:
        paths = paths[:n]
    assert paths, "no benchmark images; run scripts/setup.sh (data step)"
    return paths


def calib_images(n: int | None = None) -> list[Path]:
    """Calibration image list (train2017 subset, disjoint from val)."""
    lst = REPO / "data" / "calib_list.txt"
    paths = [REPO / p for p in lst.read_text().split()]
    if n:
        paths = paths[:n]
    assert paths, "no calibration images; run scripts/setup.sh (data step)"
    return paths


def map5095(model, data_yaml: str, imgsz: int, batch: int, device: str) -> dict:
    """mAP50-95 on the fixed educational val subset (ultralytics val)."""
    m = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device,
                  verbose=False)
    return {"map50_95": float(m.box.map), "map50": float(m.box.map50),
            "map75": float(m.box.map75)}
