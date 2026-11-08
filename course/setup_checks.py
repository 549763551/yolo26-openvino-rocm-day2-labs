"""Environment checks for YOLO26 Day 02 (AMD CPU + Radeon PRO W7900 / ROCm).

Runnable as a script (used by scripts/setup.sh) and importable from notebooks.
Performs REAL compute on both devices -- never just device detection.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _parse_cpulist(s: str) -> list[int]:
    cpus: list[int] = []
    for part in s.replace("\n", ",").split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def cgroup_cpu_quota() -> float | None:
    """Container CPU quota in cores (cpu.max), None when unlimited.

    The physical topology (e.g. 2x EPYC 9334 = 64 cores) and the container
    quota (e.g. 16 cores) are DIFFERENT resource facts and are reported
    separately — v2.0.1 fix for defect B-06."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return int(q) / int(p) if q != "max" else None
    except Exception:
        return None


def cpu_info() -> dict:
    """Full, correct CPU topology (v2.0.1 fix for defect B-06: v2.0.0
    computed `threads` as threads_per_core x sockets = 4 on this machine,
    conflated quota with cores, and omitted affinity/NUMA layout)."""
    info: dict = {}
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        for ln in lines:
            if ln.startswith("vendor_id") and "vendor" not in info:
                info["vendor"] = ln.split(":")[1].strip()
            if ln.startswith("model name") and "model" not in info:
                info["model"] = ln.split(":")[1].strip()
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        get = {}
        for ln in lscpu.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                get[k.strip()] = v.strip()
        info["sockets"] = int(get.get("Socket(s):", 0) or 0)
        info["cores_per_socket"] = int(get.get("Core(s) per socket", 0) or 0)
        info["threads_per_core"] = int(get.get("Thread(s) per core", 0) or 0)
        # v2.0.0 bug: threads = threads_per_core * sockets. Correct math:
        info["physical_cores"] = info["cores_per_socket"] * info["sockets"]
        info["logical_cpus"] = (info["physical_cores"]
                                * info["threads_per_core"])
        info["threads"] = info["logical_cpus"]        # back-compat key, now CORRECT
        info["cgroup_cpu_quota_cores"] = cgroup_cpu_quota()
        try:
            info["process_affinity_cpus"] = len(os.sched_getaffinity(0))
            info["process_affinity_sample"] = \
                ("-".join(map(str, (min(os.sched_getaffinity(0)),
                                    max(os.sched_getaffinity(0)))))
                 if len(os.sched_getaffinity(0)) > 8
                 else sorted(os.sched_getaffinity(0)))
        except Exception:
            pass
        numa = {}
        try:
            for nd in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
                cl = nd / "cpulist"
                if cl.exists():
                    numa[nd.name] = _parse_cpulist(cl.read_text().strip())
        except Exception:
            pass
        info["numa_nodes"] = len(numa) or None
        info["numa_layout"] = {k: (f"{min(v)}-{max(v)}" if len(v) > 8 else v)
                               for k, v in numa.items()}
        flags = next((ln for ln in lines if ln.startswith("flags")), "")
        have = set(flags.split())
        info["flags_sample"] = [f for f in ("avx2", "avx512f", "avx512_bf16")
                                if f in have]
        # Explicit note: AMD Zen4 has NO Intel AMX — never claim it.
        info["notes"] = ("AMD EPYC (Zen4): AVX-512/BF16 available; Intel AMX "
                         "NOT present (AMD CPU); cgroup quota is a container "
                         "limit, not the physical core count.")
    except Exception as e:  # pragma: no cover
        info["error"] = str(e)
    return info


def rocm_info() -> dict:
    out = {"gfx": None, "card_model": None, "rocm_smi_version": None}
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True).stdout
        gfx = sorted(set(__import__("re").findall(r"gfx1[0-9]{3}", r)))
        out["gfx"] = gfx
    except Exception:
        pass
    try:
        r = subprocess.run(["rocm-smi", "--showproductname", "--csv"],
                           capture_output=True, text=True).stdout
        for ln in r.splitlines():
            if "Card Model" in ln:
                out["card_model"] = ln.split(",")[1].strip()
    except Exception:
        pass
    return out


def torch_rocm_report() -> dict:
    """Versions + REAL GPU compute smoke test (matmul + torchvision NMS)."""
    import torch
    import torchvision

    rep = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
    }
    if rep["cuda_available"]:
        rep["device_name"] = torch.cuda.get_device_name(0)
        rep["device_count"] = torch.cuda.device_count()
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        c = a @ b
        torch.cuda.synchronize()
        rep["gpu_matmul_ok"] = bool(torch.isfinite(c).all().item())
        from torchvision.ops import nms
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11],
                              [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
        scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
        keep = nms(boxes, scores, 0.5)
        rep["gpu_nms_ok"] = keep.tolist() == [0, 2]
        rep["gpu_nms_result"] = keep.tolist()
    x = torch.randn(512, 512)
    rep["cpu_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    return rep


def openvino_report() -> dict:
    import openvino as ov
    core = ov.Core()
    return {"openvino": ov.__version__, "available_devices": core.available_devices}


def ultralytics_report() -> dict:
    import ultralytics
    return {"ultralytics": ultralytics.__version__}


def full_report() -> dict:
    rep = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": cpu_info(),
        "rocm": rocm_info(),
    }
    rep.update(torch_rocm_report())
    rep.update(openvino_report())
    rep.update(ultralytics_report())
    return rep


def assert_healthy(report: dict | None = None) -> None:
    """Hard gates: AMD CPU, gfx1100 GPU, ROCm torch, real compute succeeded."""
    r = report or full_report()
    assert r["cpu"]["vendor"] == "AuthenticAMD", "CPU vendor is not AMD"
    assert "gfx1100" in (r["rocm"]["gfx"] or []), "gfx1100 GPU not found"
    assert r["hip"] is not None and r["torch"].split("+")[-1].startswith("rocm"), \
        "PyTorch is not a ROCm build"
    assert r["cuda_available"], "ROCm GPU not available to PyTorch"
    assert r.get("gpu_matmul_ok") and r.get("gpu_nms_ok"), "GPU compute smoke test failed"
    assert r.get("cpu_matmul_ok"), "CPU compute smoke test failed"
    assert "CPU" in r["available_devices"], "OpenVINO has no CPU plugin"


if __name__ == "__main__":
    rep = full_report()
    rep_path = REPO / "logs" / "env_report.json"
    rep_path.parent.mkdir(exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("\n--- hard gates ---")
    try:
        assert_healthy(rep)
        print("ALL GATES PASS: AMD CPU + gfx1100 W7900 + ROCm PyTorch + OpenVINO CPU")
    except AssertionError as e:
        print(f"GATE FAILED: {e}")
        raise SystemExit(1)
