"""Lab E: serial vs asynchronous CPU+GPU pipeline, with IDENTICAL detection logic.

Both paths run the same three stages on the same images:

    read+decode+letterbox (CPU)  ->  YOLO26l forward (W7900)  ->  postprocess

Serial: stages run one-after-another per image.
Pipeline: a producer thread prepares image i+1 on the CPU while the GPU
computes image i (classic double buffering through a bounded queue).

Threads/queues/synchronization are deliberately hidden here -- students call
run_serial(...) and run_pipeline(...) and compare.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .benchmark import cuda_sync, now_ms
from .model import load_official_yolo26l


class Yolo26lStages:
    """Exposes ultralytics predictor stages so serial and pipeline share logic."""

    def __init__(self, imgsz: int = 640, warmup_image: Path | None = None):
        from .benchmark import load_bench_images
        self.imgsz = imgsz
        self.model = load_official_yolo26l()
        warm = str(warmup_image or load_bench_images(1)[0])
        self.model.predict(source=warm, imgsz=imgsz, device="cuda",
                           verbose=False)
        self.predictor = self.model.predictor

    def load(self, path: Path):
        """CPU stage 1: file read + JPEG decode."""
        return cv2.imread(str(path))

    def preprocess(self, im0) -> np.ndarray:
        """CPU stage 2: letterbox + normalize -> NCHW float32 (single or batch)."""
        ims = im0 if isinstance(im0, (list, tuple)) else [im0]
        return self.predictor.preprocess(ims)

    def infer(self, im):
        """GPU stage: YOLO26l forward on W7900."""
        return self.predictor.inference(im)

    def postprocess(self, preds, im, im0):
        """CPU/GPU stage 3: ultralytics postprocess (same as model.predict)."""
        ims = im0 if isinstance(im0, (list, tuple)) else [im0]
        batch = getattr(self.predictor, "batch", None)
        if not isinstance(batch, tuple) or len(batch[0]) != len(ims):
            self.predictor.batch = ([f"stream_{i}" for i in range(len(ims))],
                                    ims, None, "")
        return self.predictor.postprocess(preds, im, ims)


@dataclass
class PipelineRecord:
    index: int
    path: str
    t_load_ms: float = 0.0
    t_pre_ms: float = 0.0
    t_infer_ms: float = 0.0
    t_post_ms: float = 0.0
    t_e2e_ms: float = 0.0
    n_det: int = 0
    error: str | None = None
    result_ref: object = field(default=None, repr=False)
    detections: list = field(default_factory=list, repr=False)  # (cls,x1,y1,x2,y2,conf)


def _detections_of(result) -> list:
    """Extract (cls, x1, y1, x2, y2, conf) as plain floats from a Results obj."""
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    import torch
    bb = result.boxes
    xyxy = bb.xyxy.cpu() if isinstance(bb.xyxy, torch.Tensor) else bb.xyxy
    conf = bb.conf.cpu() if isinstance(bb.conf, torch.Tensor) else bb.conf
    cls = bb.cls.cpu() if isinstance(bb.cls, torch.Tensor) else bb.cls
    for i in range(len(bb)):
        out.append((int(cls[i].item()), *(float(v) for v in xyxy[i].tolist()),
                    float(conf[i].item())))
    return out


def run_serial(image_paths: list[Path], imgsz: int = 640,
               warmup: int = 3, prewarm_all: bool = True) -> dict:
    """Serial baseline: every stage sequential, one image at a time.

    prewarm_all: run one untimed pass over ALL images first so that JIT kernel
    compilation (per input shape) happens outside the measured window -- the
    same warm state for both serial and pipeline, keeping the comparison fair.
    """
    stages = Yolo26lStages(imgsz=imgsz)
    _prewarm(stages, image_paths if prewarm_all else image_paths[:warmup])
    cuda_sync()

    recs: list[PipelineRecord] = []
    t_total0 = now_ms()
    for i, p in enumerate(image_paths):
        t0 = now_ms()
        im0 = stages.load(p)
        t1 = now_ms()
        im = stages.preprocess(im0)
        t2 = now_ms()
        preds = stages.infer(im)
        cuda_sync()
        t3 = now_ms()
        res = stages.postprocess(preds, im, im0)[0]
        t4 = now_ms()
        dets = _detections_of(res)
        recs.append(PipelineRecord(i, p.name, t1 - t0, t2 - t1, t3 - t2,
                                   t4 - t3, t4 - t0, len(dets),
                                   detections=dets))
    total_ms = now_ms() - t_total0
    return _summarize("serial", recs, total_ms)


def run_pipeline(image_paths: list[Path], imgsz: int = 640,
                 queue_depth: int = 4, warmup: int = 3,
                 prewarm_all: bool = True) -> dict:
    """Async pipeline: CPU producer thread + GPU consumer thread."""
    stages = Yolo26lStages(imgsz=imgsz)
    _prewarm(stages, image_paths if prewarm_all else image_paths[:warmup])
    cuda_sync()

    n = len(image_paths)
    jobs = queue.Queue()          # (index, path, enqueue_time)
    done: list[PipelineRecord] = [None] * n   # type: ignore
    q_pre = queue.Queue(maxsize=queue_depth)  # double buffer: ready tensors
    errors: list[str] = []

    def producer() -> None:
        while True:
            item = jobs.get()
            if item is None:
                q_pre.put(None)
                return
            i, p, t_enq = item
            try:
                t0 = now_ms()
                im0 = stages.load(p)
                t1 = now_ms()
                im = stages.preprocess(im0)
                t2 = now_ms()
                q_pre.put((i, p, im0, im, t_enq, t1 - t0, t2 - t1))
            except Exception as e:                    # pragma: no cover
                errors.append(f"producer {p}: {e}")
                q_pre.put(None)

    t_total0 = now_ms()
    tp = threading.Thread(target=producer, daemon=True)
    tp.start()
    for i, p in enumerate(image_paths):
        jobs.put((i, p, now_ms()))
    jobs.put(None)

    finished = 0
    while finished < n:
        item = q_pre.get()
        if item is None:
            break
        i, p, im0, im, t_enq, t_load, t_pre = item
        t_gpu0 = now_ms()
        preds = stages.infer(im)
        cuda_sync()
        t_gpu1 = now_ms()
        res = stages.postprocess(preds, im, im0)[0]
        t_gpu2 = now_ms()
        dets = _detections_of(res)
        done[i] = PipelineRecord(i, p.name, t_load, t_pre,
                                 t_gpu1 - t_gpu0, t_gpu2 - t_gpu1,
                                 now_ms() - t_enq, len(dets),
                                 result_ref=res, detections=dets)
        finished += 1
    total_ms = now_ms() - t_total0
    tp.join(timeout=5)
    if errors:
        raise RuntimeError("; ".join(errors))
    return _summarize("async_pipeline", done, total_ms)


def _prewarm(stages: Yolo26lStages, paths: list[Path]) -> None:
    """One untimed pass so all input-shape kernels are compiled before timing."""
    for p in paths:
        im0 = stages.load(p)
        im = stages.preprocess(im0)
        stages.postprocess(stages.infer(im), im, im0)
    cuda_sync()


def _summarize(mode: str, recs: list[PipelineRecord], total_ms: float) -> dict:
    from .benchmark import pct
    ok = [r for r in recs if r and r.error is None]
    e2e = [r.t_e2e_ms for r in ok]
    return {
        "mode": mode,
        "n_images": len(ok),
        "total_ms": total_ms,
        "images_per_s": len(ok) / (total_ms / 1000.0),
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p95_ms": pct(e2e, 95),
        "e2e_note": "pipeline e2e is measured submit->result under a full "
                    "burst, so it includes queue wait; serial e2e has no queue",
        "infer_p50_ms": pct([r.t_infer_ms for r in ok], 50),
        "pre_p50_ms": pct([r.t_load_ms + r.t_pre_ms for r in ok], 50),
        "per_image": [{"i": r.index, "file": r.path, "n_det": r.n_det,
                       "load_ms": round(r.t_load_ms, 2),
                       "pre_ms": round(r.t_pre_ms, 2),
                       "infer_ms": round(r.t_infer_ms, 2),
                       "post_ms": round(r.t_post_ms, 2),
                       "e2e_ms": round(r.t_e2e_ms, 2),
                       "detections": [[d[0], *[round(v, 3) for v in d[1:5]],
                                       round(d[5], 5)] for d in r.detections]}
                      for r in ok],
    }


def _iou(a: tuple, b: tuple) -> float:
    """IoU of two (cls,x1,y1,x2,y2,conf) boxes (coordinates in slots 1..5)."""
    ax1, ay1, ax2, ay2 = a[1:5]
    bx1, by1, bx2, by2 = b[1:5]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def compare_detections(dets_a: list, dets_b: list, iou_tol: float = 0.95,
                       conf_tol: float = 0.02) -> dict:
    """One-to-one greedy parity check of two detection lists for ONE image.

    Matching rule (documented tolerance, same-model FP32 paths): a pair matches
    when class ids are EQUAL, IoU >= iou_tol and |conf_a - conf_b| <= conf_tol.
    Justification: serial and pipeline run identical weights, identical
    letterbox preprocessing, identical NMS-free postprocess; the only allowed
    difference is floating-point reduction order on the GPU, bounded well
    below these tolerances in practice. Greedy matching by best IoU.

    Returns per-image verdict with matched pairs' min IoU / max conf delta and
    the unmatched extras/misses (proof of WHAT differs, not just that)."""
    pairs = []
    for ia, a in enumerate(dets_a):
        for ib, b in enumerate(dets_b):
            if a[0] == b[0]:
                iou = _iou(a, b)
                if iou >= iou_tol and abs(a[5] - b[5]) <= conf_tol:
                    pairs.append((iou, ia, ib))
    pairs.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = []
    for iou, ia, ib in pairs:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matched.append({"iou": round(iou, 6),
                        "d_conf": round(abs(dets_a[ia][5] - dets_b[ib][5]), 8),
                        "cls": dets_a[ia][0]})
    extras = [dets_b[i] for i in range(len(dets_b)) if i not in used_b]
    misses = [dets_a[i] for i in range(len(dets_a)) if i not in used_a]
    return {
        "n_a": len(dets_a), "n_b": len(dets_b),
        "matched": len(matched),
        "min_iou": min((m["iou"] for m in matched), default=None),
        "max_d_conf": max((m["d_conf"] for m in matched), default=None),
        "extra_in_b": [[d[0], [round(v, 2) for v in d[1:5]], round(d[5], 4)]
                       for d in extras],
        "missing_in_b": [[d[0], [round(v, 2) for v in d[1:5]], round(d[5], 4)]
                         for d in misses],
        "ok": len(matched) == len(dets_a) == len(dets_b),
    }


def compare_results(serial: dict, pipeline: dict, iou_tol: float = 0.95,
                    conf_tol: float = 0.02) -> dict:
    """Per-IMAGE, per-DETECTION parity between serial and pipeline outputs.

    v2.0.1 (fixes defect B-03): v2.0.0 compared only len(boxes) per image,
    which cannot detect swapped classes, shifted boxes or changed confidence.
    This version matches every detection one-to-one (class + box IoU + conf
    tolerance) and reports exact diffs for any mismatching image.
    """
    s = {r["file"]: r.get("detections") for r in serial["per_image"]}
    p = {r["file"]: r.get("detections") for r in pipeline["per_image"]}
    files = sorted(set(s) | set(p))
    detail: dict[str, dict] = {}
    n_ok = 0
    for f in files:
        da, db = s.get(f), p.get(f)
        if da is None or db is None:
            detail[f] = {"ok": False, "error": "image missing in one output"}
            continue
        detail[f] = compare_detections(da, db, iou_tol, conf_tol)
        if detail[f]["ok"]:
            n_ok += 1
    consistent = n_ok == len(files)
    min_iou = min((d["min_iou"] for d in detail.values()
                   if d.get("min_iou") is not None), default=None)
    max_d_conf = max((d["max_d_conf"] for d in detail.values()
                      if d.get("max_d_conf") is not None), default=None)
    speedup = serial["total_ms"] / pipeline["total_ms"]
    return {
        "files_compared": len(files),
        "files_ok": n_ok,
        "consistent": consistent,
        "parity_rule": {"class": "exact", "iou_min": iou_tol,
                        "conf_abs_tol": conf_tol,
                        "match": "greedy one-to-one by IoU",
                        "justification": "same weights, same letterbox, same "
                                         "NMS-free postprocess; tolerance only "
                                         "covers FP reduction-order noise"},
        "global_min_iou": min_iou,
        "global_max_d_conf": max_d_conf,
        "mismatches": {f: d for f, d in detail.items() if not d.get("ok")},
        "serial_total_ms": serial["total_ms"],
        "pipeline_total_ms": pipeline["total_ms"],
        "throughput_gain_x": speedup,
        "serial_imgs_per_s": serial["images_per_s"],
        "pipeline_imgs_per_s": pipeline["images_per_s"],
    }
