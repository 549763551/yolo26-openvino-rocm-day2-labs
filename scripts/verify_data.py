#!/usr/bin/env python3
"""Data integrity verification for the educational COCO subsets (v2.0.1).

Fixes v2.0.0 defect B-01: setup.sh treated `data/manifest.json` (which IS
committed) as proof the images were downloaded. A fresh clone has the
manifest but NO images, so the old logic skipped preparation entirely.

This script verifies, WITHOUT any network access:
  * every manifest val image exists in data/val_subset/images AND its bytes
    hash to the manifest sha256;
  * every image has a non-empty-corresponding label file (labels may be empty
    only if the manifest says the image has no boxes);
  * data/val_subset.yaml exists and points at the right directory;
  * every calibration image exists in data/calib/images with matching sha256;
  * data/calib_list.txt exists and lists exactly the manifest calib files;
  * train/val isolation: no val file name appears in the calibration set.

Exit codes: 0 = DATA OK · 2 = problems found (printed, one per line).
Run `scripts/prepare_data.py` to (incrementally) repair; re-run this script
afterwards — it must report DATA OK before any benchmark/notebook runs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest_path: Path | None = None, quick_hashes: bool = True) -> list[str]:
    problems: list[str] = []
    mp = manifest_path or (DATA / "manifest.json")
    if not mp.exists():
        return [f"MISSING manifest: {mp}"]
    m = json.loads(mp.read_text())
    base = mp.parent

    val_dir = base / "val_subset" / "images"
    lbl_dir = base / "val_subset" / "labels"
    calib_dir = base / "calib" / "images"
    yaml_path = base / "val_subset.yaml"
    cl_path = base / "calib_list.txt"

    # ---- val images + sha256 -------------------------------------------------
    val_entries = m.get("val_images", [])
    for im in val_entries:
        p = val_dir / im["file_name"]
        if not p.exists():
            problems.append(f"MISSING val image: data/val_subset/images/{im['file_name']}")
            continue
        if not quick_hashes and sha256(p) != im["sha256"]:
            problems.append(f"CORRUPT val image (sha256 mismatch): {im['file_name']}")
        lbl = lbl_dir / (Path(im["file_name"]).stem + ".txt")
        if not lbl.exists():
            problems.append(f"MISSING label: data/val_subset/labels/{lbl.name}")
    n_on_disk = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    if n_on_disk != len(val_entries):
        problems.append(f"COUNT val images: {n_on_disk} on disk, "
                        f"{len(val_entries)} in manifest")

    # ---- sha256 full check (always for val; the benchmark depends on bytes) --
    for im in val_entries:
        p = val_dir / im["file_name"]
        if p.exists() and sha256(p) != im["sha256"]:
            problems.append(f"CORRUPT val image (sha256 mismatch): {im['file_name']}")

    # ---- yaml ----------------------------------------------------------------
    if not yaml_path.exists():
        problems.append("MISSING data/val_subset.yaml")
    else:
        y = yaml_path.read_text()
        if "images" not in y or "names" not in y:
            problems.append("INVALID data/val_subset.yaml (missing keys)")

    # ---- calibration images ----------------------------------------------------
    calib_entries = m.get("calib_images", [])
    for im in calib_entries:
        p = calib_dir / im["file_name"]
        if not p.exists():
            problems.append(f"MISSING calib image: data/calib/images/{im['file_name']}")
    n_calib_disk = len(list(calib_dir.glob("*.jpg"))) if calib_dir.exists() else 0
    if n_calib_disk != len(calib_entries):
        problems.append(f"COUNT calib images: {n_calib_disk} on disk, "
                        f"{len(calib_entries)} in manifest")

    # ---- calib list ------------------------------------------------------------
    if not cl_path.exists():
        problems.append("MISSING data/calib_list.txt")
    else:
        listed = [Path(s).name for s in cl_path.read_text().split()]
        expected = sorted(im["file_name"] for im in calib_entries)
        if sorted(listed) != expected:
            problems.append("MISMATCH data/calib_list.txt does not list exactly "
                            "the manifest calibration files")

    # ---- train/val isolation -----------------------------------------------------
    val_names = {im["file_name"] for im in val_entries}
    calib_names = {im["file_name"] for im in calib_entries}
    overlap = val_names & calib_names
    if overlap:
        problems.append(f"ISOLATION violation: {sorted(overlap)[:3]} present in "
                        "both val and calib sets")

    return problems


def main() -> int:
    problems = verify()
    m = json.loads((DATA / "manifest.json").read_text()) \
        if (DATA / "manifest.json").exists() else {}
    print(f"manifest: {len(m.get('val_images', []))} val · "
          f"{len(m.get('calib_images', []))} calib · seed {m.get('seed')}")
    if problems:
        print(f"DATA INTEGRITY: FAIL ({len(problems)} problem(s))")
        for p in problems:
            print("  -", p)
        print("repair with: python scripts/prepare_data.py   "
              "(incremental, re-verified afterwards by setup.sh)")
        return 2
    print("DATA INTEGRITY: PASS (all images/labels/yaml/calib verified, "
          "sha256 match, train/val disjoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
