"""Unit tests (fast, no GPU / no model weights required).

These run in GitHub Actions CI on any CPU; hardware-dependent integration
tests live in tests/test_integration_amd.py and are skipped unless the
AMD + gfx1100 environment is present.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_course_importable():
    import course.benchmark  # noqa: F401
    import course.model      # noqa: F401
    import course.charts     # noqa: F401


def test_percentile():
    from course.benchmark import pct
    s = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pct(s, 50) == 3.0
    assert pct(s, 0) == 1.0
    assert pct(s, 100) == 5.0


def test_stats_of():
    from course.benchmark import stats_of
    st = stats_of([10.0] * 10)
    assert st["n"] == 10
    assert st["mean_ms"] == 10.0
    assert st["p50_ms"] == 10.0


def test_bench_excludes_warmup():
    from course.benchmark import bench
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
    samples = bench(fn, warmup=3, iters=5, sync=False)
    assert len(samples) == 5 and calls["n"] == 8


def test_identity_record_shape():
    from course.model import OFFICIAL_PT
    if not OFFICIAL_PT.exists():      # CI has no weights (never committed)
        pytest.skip("weights not present (CI)")
    from course.model import identity_record
    rec = identity_record()
    assert rec["verified"]["num_classes"] == 80
    assert rec["verified"]["task"] == "detect"
    assert len(rec["sha256"]) == 64


def test_coco_to_yolo(tmp_path):
    from scripts.prepare_data import coco_to_yolo_labels
    ann = {
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}],
        "images": [{"id": 1, "width": 100, "height": 50}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "iscrowd": 0},
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 50, 50], "iscrowd": 0},
        ],
    }
    out = coco_to_yolo_labels(ann, {1})
    assert out[1][0].startswith("0 0.200000 0.300000")   # person: cx=20/100, cy=15/50
    assert out[1][1].startswith("1 0.250000 0.500000")   # bicycle: cx=25/100, cy=25/50
