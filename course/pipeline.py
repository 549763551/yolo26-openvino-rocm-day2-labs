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


def _detections_of(result) -> int:
    return int(len(result.boxes)) if result.boxes is not None else 0


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
        recs.append(PipelineRecord(i, p.name, t1 - t0, t2 - t1, t3 - t2,
                                   t4 - t3, t4 - t0, _detections_of(res)))
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
        done[i] = PipelineRecord(i, p.name, t_load, t_pre,
                                 t_gpu1 - t_gpu0, t_gpu2 - t_gpu1,
                                 now_ms() - t_enq, _detections_of(res),
                                 result_ref=res)
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
                       "e2e_ms": round(r.t_e2e_ms, 2)} for r in ok],
    }


def compare_results(serial: dict, pipeline: dict, tol_det: int = 0) -> dict:
    """Same images + same logic -> detection counts must match (tolerance 0)."""
    s = {r["file"]: r["n_det"] for r in serial["per_image"]}
    p = {r["file"]: r["n_det"] for r in pipeline["per_image"]}
    mismatch = {f: (s[f], p[f]) for f in s if s[f] != p.get(f)}
    speedup = serial["total_ms"] / pipeline["total_ms"]
    return {
        "files_compared": len(s),
        "detection_mismatches": mismatch,
        "consistent": all(s[f] == p.get(f) for f in s),
        "serial_total_ms": serial["total_ms"],
        "pipeline_total_ms": pipeline["total_ms"],
        "throughput_gain_x": speedup,
        "serial_imgs_per_s": serial["images_per_s"],
        "pipeline_imgs_per_s": pipeline["images_per_s"],
    }
