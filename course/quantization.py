"""NNCF INT8 post-training quantization of the official YOLO26l OpenVINO model.

Calibration uses REAL COCO train2017 images (data/calib/images), which are
disjoint from the val2017 subset used for benchmarking / mAP.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent


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


def calibration_source(n_images: int | None = None, imgsz: int = 640):
    """Yields raw BGR images read from the fixed calibration list."""
    import cv2
    from .benchmark import calib_images
    transform = calibration_transform(imgsz)

    def gen() -> Iterator:
        for p in calib_images(n_images):
            im = cv2.imread(str(p))
            if im is not None:
                yield im
    return gen, transform


def quantize_int8(n_calib_images: int | None = None, imgsz: int = 640,
                  subset_size: int = 150, verbose: bool = True) -> Path:
    """Quantize OpenVINO FP32 IR -> INT8 with NNCF post-training quantization.

    Recipe: NNCF `MIXED` preset on REAL COCO train2017 calibration images
    (default 150, disjoint from the val subset). On this machine MIXED kept
    clearly more mAP than the plain PERFORMANCE preset at the same latency
    (measured: see results/latest.json labC) -- that comparison is part of the
    course's honest-results policy.

    Returns the INT8 model directory (ultralytics-loadable, metadata copied).
    """
    import nncf
    import openvino as ov
    from .model import openvino_fp32_dir, openvino_int8_dir

    fp32_dir = openvino_fp32_dir()
    xml = fp32_dir / "yolo26l.xml"
    assert xml.exists(), "FP32 export missing; run course.model.export_openvino_fp32()"
    int8_dir = openvino_int8_dir()
    if (int8_dir / "yolo26l.xml").exists():
        return int8_dir

    core = ov.Core()
    ov_model = core.read_model(str(xml))
    gen, transform = calibration_source(n_calib_images, imgsz)
    input_name = ov_model.inputs[0].any_name

    def transform_fn(image_bgr):
        return {input_name: transform(image_bgr)}

    dataset = nncf.Dataset(gen(), transform_fn)
    if verbose:
        print(f"[quantize] NNCF {nncf.__version__} · preset MIXED · "
              f"subset_size={subset_size} · real COCO train2017 images")
    quantized = nncf.quantize(ov_model, dataset, subset_size=subset_size,
                              preset=nncf.QuantizationPreset.MIXED)

    int8_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(int8_dir / "yolo26l.xml"),
                  compress_to_fp16=False)
    for meta in fp32_dir.glob("metadata*"):
        shutil.copyfile(meta, int8_dir / meta.name)
    if verbose:
        mb = (int8_dir / "yolo26l.bin").stat().st_size / 1e6
        print(f"[quantize] saved INT8 IR -> {int8_dir}  ({mb:.1f} MB weights)")
    return int8_dir


def _val_image_paths(n: int):
    from .benchmark import load_bench_images
    return load_bench_images(n)


def model_dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.iterdir() if f.suffix in (".xml", ".bin")) / 1e6
