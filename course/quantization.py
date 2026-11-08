"""NNCF INT8 post-training quantization of the official YOLO26l OpenVINO model.

Calibration uses REAL COCO train2017 images (data/calib/images), which are
disjoint from the val2017 subset used for benchmarking / mAP.

v2.0.1 (fixes defect B-05): every INT8 cache entry now carries a full
IDENTITY KEY —
  * official yolo26l.pt sha256;
  * OpenVINO FP32 export xml+bin sha256;
  * calibration dataset identity (ordered file list + per-file sha256 from
    data/manifest.json) and the exact number of calibration samples used;
  * NNCF preset + version, subset_size, imgsz and export config.
Changing ANY of these (e.g. STUDENT_CALIB_N 150 -> 30) produces a different
key and therefore a DIFFERENT cache directory — a stale 150-image model can
never be silently served for a 30-image calibration request. Multiple
calibration sizes coexist in weights/int8_cache/<key>/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
CACHE_ROOT = REPO / "weights" / "int8_cache"


def calibration_transform(imgsz: int = 640):
    """Ultralytics-style letterbox preprocessing identical to predict-time."""
    import numpy as np
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(imgsz, auto=False, stride=32)

    def transform(image_bgr: "np.ndarray") -> "np.ndarray":
        im = letterbox(image=image_bgr)
        im = im[:, :, ::-1].transpose(2, 0, 1)      # BGR->RGB, HWC->CHW
        im = np.ascontiguousarray(im).astype(np.float32) / 255.0
        return im[None]                             # 1x3xHxW
    return transform


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_identity(n_images: int | None) -> dict:
    """Ordered calibration file list + per-file sha256 (from data manifest).

    The first n_images entries of data/calib_list.txt are the exact files a
    calibration run consumes — they define the dataset identity."""
    from .benchmark import calib_images
    paths = calib_images(n_images)
    manifest = json.loads((REPO / "data" / "manifest.json").read_text())
    by_name = {c["file_name"]: c["sha256"]
               for c in manifest.get("calib_images", [])}
    files = [{"name": p.name, "sha256": by_name[p.name]} for p in paths]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"n_calib": len(files), "files": files,
            "dataset_sha256": digest}


def cache_key(n_calib_images: int | None, imgsz: int, subset_size: int,
              preset: str) -> tuple[str, dict]:
    """Full identity of an INT8 artifact (see module docstring)."""
    from . import model as M
    import nncf  # delayed: only needed when actually building/validating

    fp32_dir = M.openvino_fp32_dir()
    ident = {
        "pt_sha256": M.model_sha256(),
        "fp32_xml_sha256": _sha256(fp32_dir / "yolo26l.xml"),
        "fp32_bin_sha256": _sha256(fp32_dir / "yolo26l.bin"),
        "calibration": calibration_identity(n_calib_images),
        "subset_size": subset_size,
        "preset": preset,
        "imgsz": imgsz,
        "nncf_version": nncf.__version__,
        "quantizer": "nncf.quantize",
        "export": {"dynamic": False, "batch": 1, "compress_to_fp16": False},
    }
    key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return key, ident


def int8_cache_dir(n_calib_images: int | None = None, imgsz: int = 640,
                   subset_size: int = 150, preset: str = "mixed") -> Path:
    """Cache directory for this exact configuration (key-identity path).

    Name carries the `_openvino_model` suffix so ultralytics detects the
    format from the directory name."""
    key, _ = cache_key(n_calib_images, imgsz, subset_size, preset)
    return CACHE_ROOT / f"yolo26l_int8_{preset}_openvino_model_{key[:16]}"


def quantize_int8(n_calib_images: int | None = None, imgsz: int = 640,
                  subset_size: int = 150, preset: str = "mixed",
                  verbose: bool = True, force: bool = False) -> Path:
    """Quantize OpenVINO FP32 IR -> INT8 with NNCF post-training quantization.

    Cache behaviour (v2.0.1): a cache hit requires the stored CACHE_KEY.json
    to match the CURRENT configuration identity byte-for-byte (model, FP32
    export, calibration files+count, preset, subset_size, imgsz, nncf
    version). Any change re-quantizes into a new directory.

    Returns the INT8 model directory (ultralytics-loadable, metadata copied).
    """
    import nncf
    import openvino as ov
    from . import model as M

    fp32_dir = M.openvino_fp32_dir()
    xml = fp32_dir / "yolo26l.xml"
    assert xml.exists(), "FP32 export missing; run course.model.export_openvino_fp32()"

    key, ident = cache_key(n_calib_images, imgsz, subset_size, preset)
    int8_dir = CACHE_ROOT / f"yolo26l_int8_{preset}_openvino_model_{key[:16]}"
    key_file = int8_dir / "CACHE_KEY.json"
    if (not force and key_file.exists()
            and (int8_dir / "yolo26l.xml").exists()
            and (int8_dir / "yolo26l.bin").exists()):
        stored = json.loads(key_file.read_text())
        if stored.get("key") == key:
            if verbose:
                print(f"[quantize] cache HIT (identity verified): {int8_dir.name} "
                      f"· n_calib={ident['calibration']['n_calib']} preset={preset}")
            return int8_dir
        if verbose:
            print("[quantize] cache key mismatch -> re-quantizing")
    if verbose:
        print(f"[quantize] building NEW INT8 model · key {key[:24]} · "
              f"n_calib={ident['calibration']['n_calib']} · preset={preset} · "
              f"NNCF {nncf.__version__}")

    core = ov.Core()
    ov_model = core.read_model(str(xml))

    from .benchmark import calib_images
    import cv2
    transform = calibration_transform(imgsz)
    paths = calib_images(n_calib_images)

    def gen() -> Iterator:
        for p in paths:
            im = cv2.imread(str(p))
            if im is not None:
                yield im

    input_name = ov_model.inputs[0].any_name

    def transform_fn(image_bgr):
        return {input_name: transform(image_bgr)}

    dataset = nncf.Dataset(gen(), transform_fn)
    t0 = time.perf_counter()
    preset_enum = getattr(nncf.QuantizationPreset, preset.upper())
    quantized = nncf.quantize(ov_model, dataset, subset_size=subset_size,
                              preset=preset_enum)
    build_s = time.perf_counter() - t0

    int8_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(int8_dir / "yolo26l.xml"),
                  compress_to_fp16=False)
    for meta in fp32_dir.glob("metadata*"):
        shutil.copyfile(meta, int8_dir / meta.name)
    record = {
        "key": key,
        "identity": ident,
        "quantization_time_s": round(build_s, 1),
        "calibration_manifest": ident["calibration"]["files"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifacts": {
            "xml_sha256": _sha256(int8_dir / "yolo26l.xml"),
            "bin_sha256": _sha256(int8_dir / "yolo26l.bin"),
        },
    }
    key_file.write_text(json.dumps(record, indent=2))
    if verbose:
        mb = (int8_dir / "yolo26l.bin").stat().st_size / 1e6
        print(f"[quantize] saved INT8 IR -> {int8_dir}  ({mb:.1f} MB weights, "
              f"{build_s:.0f}s) · identity recorded in CACHE_KEY.json")
    return int8_dir


def model_dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.iterdir()
               if f.suffix in (".xml", ".bin")) / 1e6
