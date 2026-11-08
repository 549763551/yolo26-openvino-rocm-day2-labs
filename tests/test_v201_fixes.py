"""Unit tests for the v2.0.1 fixes (CI-safe: no GPU, no weights, no network).

Covers:
  * R1.1 verify_data: manifest-only clone must FAIL (defect B-01 regression)
  * R1.2 evidence chain: summary json hash == file bytes; SHA256SUMS cover
    files; tampering is detected (defect B-02)
  * R1.3 per-detection parity: class/box/conf matching (defect B-03)
  * R2.2 CPU topology math (defect B-06)
  * R1.4 static model gate patterns
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ------------------------------------------------------------------ R1.1 ----
def _make_data_dir(tmp_path: Path, n_val=2, n_calib=2, complete=True,
                   corrupt_one=False):
    d = tmp_path / "data"
    (d / "val_subset" / "images").mkdir(parents=True)
    (d / "val_subset" / "labels").mkdir(parents=True)
    (d / "calib" / "images").mkdir(parents=True)
    manifest = {"seed": 42, "val_images": [], "calib_images": []}
    for i in range(n_val):
        fn = f"val{i:012d}.jpg"
        manifest["val_images"].append({"file_name": fn, "sha256": "x" * 64})
        if complete:
            (d / "val_subset" / "images" / fn).write_bytes(b"fakejpg")
            (d / "val_subset" / "labels" / f"val{i:012d}.txt").write_text("")
    for i in range(n_calib):
        fn = f"cal{i:012d}.jpg"
        manifest["calib_images"].append({"file_name": fn, "sha256": "y" * 64})
        if complete:
            (d / "calib" / "images" / fn).write_bytes(b"fakejpg")
    # real sha256 for present files
    for section, folder in (("val_images", "val_subset/images"),
                            ("calib_images", "calib/images")):
        for im in manifest[section]:
            p = d / folder / im["file_name"]
            if p.exists():
                im["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    if complete:
        (d / "val_subset.yaml").write_text("path: x\nval: images\nnames: {0: a}\n")
        (d / "calib_list.txt").write_text(
            "\n".join(f"data/calib/images/{im['file_name']}"
                      for im in manifest["calib_images"]) + "\n")
    if corrupt_one:
        p = d / "val_subset" / "images" / manifest["val_images"][0]["file_name"]
        if p.exists():
            p.write_bytes(b"CORRUPTED")
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_verify_data_pass_complete(tmp_path):
    from scripts.verify_data import verify
    d = _make_data_dir(tmp_path, complete=True)
    assert verify(d / "manifest.json") == []


def test_verify_data_fails_manifest_only(tmp_path):
    """The exact v2.0.0 fresh-clone defect: manifest present, images absent."""
    from scripts.verify_data import verify
    d = _make_data_dir(tmp_path, complete=False)
    problems = verify(d / "manifest.json")
    assert any("MISSING val image" in p for p in problems)
    assert any("MISSING calib image" in p for p in problems)
    assert any("COUNT val images" in p for p in problems)
    assert any("MISSING data/val_subset.yaml" in p for p in problems)
    assert any("MISSING data/calib_list.txt" in p for p in problems)
    assert problems, "manifest-only must NOT pass"


def test_verify_data_fails_on_corruption(tmp_path):
    from scripts.verify_data import verify
    d = _make_data_dir(tmp_path, complete=True, corrupt_one=True)
    problems = verify(d / "manifest.json")
    assert any("CORRUPT" in p for p in problems)


# ------------------------------------------------------------------ R1.2 ----
def test_save_run_evidence_chain(tmp_path, monkeypatch):
    from course import benchmark as B
    monkeypatch.setattr(B, "RESULTS", tmp_path)
    meta = B.RunMeta(experiment="t_exp", variant="t_var", device="cpu",
                     runtime="test", warmup=2, iters=5)
    summary = B.save_run(meta, [10.0, 11.0, 12.0, 13.0, 14.0])
    exp_dir = tmp_path / "t_exp"
    jf = exp_dir / f"{summary['run_id']}_summary.json"
    # the v2.0.0 defect: recorded json hash != file bytes after rewrite.
    actual = hashlib.sha256(jf.read_bytes()).hexdigest()
    assert summary["evidence"]["json_sha256"] == actual
    # CSV sha also matches bytes
    cf = exp_dir / f"{summary['run_id']}.csv"
    assert summary["evidence"]["csv_sha256"] == \
        hashlib.sha256(cf.read_bytes()).hexdigest()
    # SHA256SUMS covers both and verifies
    sums = (exp_dir / "SHA256SUMS").read_text()
    assert summary["run_id"] + ".csv" in sums
    assert summary["run_id"] + "_summary.json" in sums


def test_tamper_detected(tmp_path, monkeypatch):
    from scripts import verify_evidence as VE
    from course import benchmark as B
    monkeypatch.setattr(B, "RESULTS", tmp_path)
    meta = B.RunMeta(experiment="t2", variant="v", device="cpu",
                     runtime="test", warmup=1, iters=4)
    s = B.save_run(meta, [1.0, 2.0, 3.0, 4.0])
    problems = []
    VE.verify_dir_sums(tmp_path / "t2", problems)
    VE.verify_summary(tmp_path / "t2" / f"{s['run_id']}_summary.json", problems)
    assert problems == []
    # tamper: change one CSV byte -> both dir sums and summary checks fail
    cf = tmp_path / "t2" / f"{s['run_id']}.csv"
    cf.write_text(cf.read_text().replace("2.0000", "9.9999"))
    problems = []
    VE.verify_dir_sums(tmp_path / "t2", problems)
    VE.verify_summary(tmp_path / "t2" / f"{s['run_id']}_summary.json", problems)
    assert any("TAMPERED" in p or "MISMATCH" in p for p in problems)
    # tamper: rewrite summary stats -> stats mismatch detected
    jf = tmp_path / "t2" / f"{s['run_id']}_summary.json"
    cf.write_text(cf.read_text())  # restore csv bytes
    d = json.loads(jf.read_text())
    d["stats"]["p50_ms"] = 12345.0
    jf.write_text(json.dumps(d))
    problems = []
    VE.verify_summary(jf, problems)
    assert problems


# ------------------------------------------------------------------ R1.3 ----
def _det(cls=0, x=0, y=0, w=10, h=10, conf=0.9):
    return (cls, float(x), float(y), float(x + w), float(y + h), conf)


def test_parity_identical():
    from course.pipeline import compare_detections
    a = [_det(), _det(cls=1, x=50), _det(cls=2, y=77, conf=0.4)]
    r = compare_detections(a, list(a))
    assert r["ok"] and r["matched"] == 3 and r["min_iou"] == 1.0


def test_parity_swapped_class_detected():
    from course.pipeline import compare_detections
    a = [_det(cls=0, x=0), _det(cls=1, x=50)]
    b = [_det(cls=1, x=0), _det(cls=0, x=50)]  # same boxes, swapped classes
    r = compare_detections(a, b)
    assert not r["ok"]
    assert r["extra_in_b"] and r["missing_in_b"]


def test_parity_box_shift_within_tolerance():
    from course.pipeline import compare_detections
    a = [_det(x=0, w=100, h=100)]
    b = [_det(x=1, w=100, h=100)]      # IoU ~0.96 >= 0.95
    r = compare_detections(a, b)
    assert r["ok"], "1px shift should match within documented tolerance"


def test_parity_box_shift_beyond_tolerance():
    from course.pipeline import compare_detections
    a = [_det(x=0, w=40, h=40)]
    b = [_det(x=10, w=40, h=40)]       # IoU ~0.475 < 0.95
    r = compare_detections(a, b)
    assert not r["ok"]


def test_parity_conf_drift_detected():
    from course.pipeline import compare_detections
    a = [_det(conf=0.90)]
    b = [_det(conf=0.60)]              # way beyond 0.02 tolerance
    r = compare_detections(a, b)
    assert not r["ok"]


def test_parity_count_only_check_would_have_passed():
    """Regression proof for defect B-03: counts equal, contents differ."""
    from course.pipeline import compare_detections
    a = [_det(cls=0, x=0), _det(cls=15, x=100, conf=0.8)]
    b = [_det(cls=7, x=0, conf=0.3), _det(cls=3, x=200, y=200, conf=0.1)]
    assert len(a) == len(b)            # v2.0.0 check: PASS (wrongly)
    r = compare_detections(a, b)
    assert not r["ok"]                 # v2.0.1 check: correctly FAIL


# ------------------------------------------------------------------ R2.2 ----
def test_cpu_topology_math():
    from course.setup_checks import cpu_info
    info = cpu_info()
    if info.get("error"):
        pytest.skip("no /proc access")
    assert info["logical_cpus"] == (info["cores_per_socket"]
                                    * info["sockets"]
                                    * info["threads_per_core"])
    assert info["physical_cores"] == info["cores_per_socket"] * info["sockets"]
    assert "amx" not in info["flags_sample"]
    assert "AMX NOT present" in info["notes"]
    # container quota is reported separately from physical cores
    q = info.get("cgroup_cpu_quota_cores")
    assert q is None or q != info["physical_cores"] or q is None


# ------------------------------------------------------------------ R1.4 ----
def test_static_model_gate():
    from scripts.execute_notebooks import static_model_gate, NOTEBOOKS
    for nb in NOTEBOOKS:
        src = "\n".join("".join(c["source"]) for c in json.loads(
            (REPO / "notebooks" / nb).read_text())["cells"]
            if c["cell_type"] == "code")
        assert static_model_gate(nb, src) == [], f"model gate failed for {nb}"
    good = ("from course.model import load_official_yolo26l\n"
            "m = load_official_yolo26l()")
    assert static_model_gate("t", good) == []
    for bad in ("YOLO('yolo26n.pt')", "YOLO('best.pt')",
                "boxes = torch.randn(3,4)"):
        assert static_model_gate("t", bad), f"gate missed {bad!r}"
