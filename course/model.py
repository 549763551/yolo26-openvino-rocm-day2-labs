"""Official YOLO26l model identity, loading and OpenVINO export helpers.

Hard rule of this course: the ONLY weights allowed are the official
Ultralytics YOLO26l COCO detection weights (80 classes). Every benchmark in
every notebook must share this single model identity, recorded in
weights/model_identity.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "weights"
OFFICIAL_PT = WEIGHTS / "yolo26l.pt"
EXPECTED_CLASSES = 80
EXPECTED_TASK = "detect"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official_yolo26l():
    """Load the official YOLO26l detection model and verify its identity."""
    from ultralytics import YOLO
    assert OFFICIAL_PT.exists(), (
        f"{OFFICIAL_PT} missing - run scripts/setup.sh first")
    model = YOLO(str(OFFICIAL_PT))
    _verify_identity(model)
    return model


def _verify_identity(model) -> dict:
    ident = identity_record()
    assert len(model.names) == EXPECTED_CLASSES, (
        f"expected {EXPECTED_CLASSES} COCO classes, got {len(model.names)}")
    task = str(getattr(model.model, "task", "")).lower()
    assert task == EXPECTED_TASK, f"expected detection task, got {task}"
    return ident


def identity_record() -> dict:
    """Return (and cache) the recorded identity of weights/yolo26l.pt."""
    cache = WEIGHTS / "model_identity.json"
    if cache.exists():
        rec = json.loads(cache.read_text())
        if "verified" not in rec:          # normalize the setup.sh flat record
            rec["verified"] = {"num_classes": rec.get("num_classes", 80),
                               "task": rec.get("task", "detect"),
                               "dataset": "COCO 80-class object detection"}
        return rec
    rec = {
        "file": str(OFFICIAL_PT),
        "size_bytes": OFFICIAL_PT.stat().st_size,
        "sha256": sha256_of(OFFICIAL_PT),
        "source": "official Ultralytics YOLO('yolo26l.pt') release download",
        "verified": {
            "num_classes": EXPECTED_CLASSES,
            "task": EXPECTED_TASK,
            "dataset": "COCO 80-class object detection",
        },
    }
    WEIGHTS.mkdir(exist_ok=True)
    cache.write_text(json.dumps(rec, indent=2))
    return rec


def model_sha256() -> str:
    return identity_record()["sha256"]


def export_openvino_fp32(imgsz: int = 640) -> Path:
    """Export the SAME official yolo26l.pt to OpenVINO FP32 IR.

    Returns the export directory, e.g. weights/yolo26l_openvino_model/
    """
    from ultralytics import YOLO
    out = WEIGHTS / "yolo26l_openvino_model"
    if (out / "yolo26l.xml").exists():
        return out
    model = YOLO(str(OFFICIAL_PT))
    path = model.export(format="openvino", imgsz=imgsz, dynamic=False,
                        int8=False, batch=1)
    exported = Path(path).parent if str(path).endswith(".xml") else Path(path)
    if exported != out and exported.exists():
        out.mkdir(parents=True, exist_ok=True)
        for f in exported.iterdir():
            (out / f.name).write_bytes(f.read_bytes())
    assert (out / "yolo26l.xml").exists(), "OpenVINO FP32 export failed"
    return out


def export_onnx(imgsz: int = 640) -> Path:
    """Optional ONNX export (used only for inspection / debugging)."""
    from ultralytics import YOLO
    out = WEIGHTS / "yolo26l.onnx"
    if out.exists():
        return out
    model = YOLO(str(OFFICIAL_PT))
    path = model.export(format="onnx", imgsz=imgsz, dynamic=False)
    p = Path(path)
    if p != out and p.exists():
        out.write_bytes(p.read_bytes())
    return out


def openvino_fp32_dir() -> Path:
    return WEIGHTS / "yolo26l_openvino_model"


def openvino_int8_dir(n_calib_images: int | None = None, imgsz: int = 640,
                      subset_size: int = 150, preset: str = "mixed") -> Path:
    """INT8 cache directory for the given calibration configuration (v2.0.1).

    Delegates to course.quantization.int8_cache_dir so that every consumer
    (notebooks, experiments, tests) resolves the SAME identity-keyed path.
    The v2.0.0 fixed path weights/yolo26l_int8_openvino_model is no longer
    used implicitly — see course/quantization.py docstring."""
    from .quantization import int8_cache_dir
    return int8_cache_dir(n_calib_images, imgsz, subset_size, preset)
