"""Environment checks for YOLO26 Day 02 (AMD CPU + Radeon PRO W7900 / ROCm).

Runnable as a script (used by scripts/setup.sh) and importable from notebooks.
Performs REAL compute on both devices -- never just device detection.
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cpu_info() -> dict:
    info = {"vendor": None, "model": None, "physical_cores": None,
            "threads": None, "sockets": None, "numa_nodes": None,
            "flags_sample": []}
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        for ln in lines:
            if ln.startswith("vendor_id") and info["vendor"] is None:
                info["vendor"] = ln.split(":")[1].strip()
            if ln.startswith("model name") and info["model"] is None:
                info["model"] = ln.split(":")[1].strip()
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for ln in lscpu.splitlines():
            for key, tag in [("physical_cores", "Core(s) per socket"),
                             ("threads", "Thread(s) per core"), ("sockets", "Socket(s):"),
                             ("numa_nodes", "NUMA node(s):")]:
                if ln.startswith(tag):
                    info[key] = int(ln.split(":")[1].strip())
        if info["physical_cores"] and info["sockets"]:
            info["physical_cores"] *= info["sockets"]
        if info["threads"] and info["physical_cores"]:
            info["threads"] = info["threads"] * info["sockets"]
        flags = next((ln for ln in lines if ln.startswith("flags")), "")
        info["flags_sample"] = [f for f in ("avx2", "avx512f", "avx512_bf16", "amx")
                                if f in flags.split()]
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
