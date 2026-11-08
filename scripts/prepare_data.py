#!/usr/bin/env python3
"""Prepare educational COCO subsets for YOLO26 Day 02.

Selection (fixed, reproducible):
  - benchmark / mAP subset: N_VAL images from COCO val2017
  - INT8 calibration subset : N_CALIB images from COCO train2017
  - the two subsets NEVER overlap (different COCO splits by construction)

Transport: the official host images.cocodataset.org is tried first. Some lab
networks (e.g. allow-listed proxies) block it; in that case we fall back to
HuggingFace CDN mirrors that carry the ORIGINAL COCO 2017 files:

  - official annotation JSONs  -> HF dataset sashankSaya/coco2017 (byte-exact
    official instances_train2017.json / instances_val2017.json)
  - val2017 images             -> HF dataset Selenophili/val2017 (val2017.zip)
  - train2017 images           -> HF dataset danasone/coco2017 (train2017.zip,
    read REMOTELY via HTTP range requests - only the selected files are
    transferred, never the 19 GB archive)

Integrity: every extracted val image is verified against the official
annotation metadata (exact file_name + width + height). Outputs carry a
manifest with ids, sha256, sources and COCO/Flickr license records.

This is an *educational subset*, not an official COCO benchmark.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DL = DATA / "downloads"
N_VAL = 120     # benchmark / accuracy subset (val2017)
N_CALIB = 150   # INT8 calibration subset (train2017)
SEED = 42

OFFICIAL_ANN = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OFFICIAL_IMG = "https://images.cocodataset.org/{split}/{fname}"
HF = {
    "ann_val": "https://huggingface.co/datasets/sashankSaya/coco2017/resolve/main/annotations/instances_val2017.json",
    "ann_train": "https://huggingface.co/datasets/sashankSaya/coco2017/resolve/main/annotations/instances_train2017.json",
    "val_zip": "https://huggingface.co/datasets/Selenophili/val2017/resolve/main/val2017.zip",
    "train_zip": "https://huggingface.co/datasets/danasone/coco2017/resolve/main/train2017.zip",
}
TRAIN_SOURCE = "COCO train2017 (original files via HF mirror danasone/coco2017, HTTP-range remote zip)"
VAL_SOURCE = "COCO val2017 (original files; official host or HF mirror Selenophili/val2017)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(url: str, dest: Path, max_time: int = 1800) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url.rsplit('/', 1)[-1]} ({url.split('/')[2]})")
    r = subprocess.run(["curl", "-L", "-sS", "--retry", "5", "--retry-all-errors",
                        "--max-time", str(max_time), "-o", str(dest) + ".part", url])
    if r.returncode != 0:
        dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {url}")
    (str(dest) + ".part") and Path(str(dest) + ".part").rename(dest)


def official_host_reachable() -> bool:
    try:
        req = urllib.request.Request(OFFICIAL_IMG.format(split="val2017",
                                fname="000000397133.jpg"), method="HEAD")
        urllib.request.urlopen(req, timeout=8)
        return True
    except Exception:
        return False


def pick(images: list[dict], n: int) -> list[dict]:
    """Deterministic selection of n images, spread over the sorted id space."""
    rng = random.Random(SEED)
    chosen = rng.sample(sorted(images, key=lambda im: im["id"]), n)
    return sorted(chosen, key=lambda im: im["id"])


def load_annotations(use_official: bool) -> tuple[dict, dict]:
    val_p = DL / "instances_val2017.json"
    train_p = DL / "instances_train2017.json"
    if not val_p.exists():
        if use_official:
            ann_zip = DL / "annotations_trainval2017.zip"
            curl(OFFICIAL_ANN, ann_zip)
            with zipfile.ZipFile(ann_zip) as z:
                for name, out in [("annotations/instances_val2017.json", val_p),
                                  ("annotations/instances_train2017.json", train_p)]:
                    with z.open(name) as fsrc, open(DL / Path(name).name, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
        else:
            curl(HF["ann_val"], val_p)
            curl(HF["ann_train"], train_p)
    return (json.loads(val_p.read_text()), json.loads(train_p.read_text()))


def get_val_images(val_imgs: list[dict], use_official: bool) -> None:
    """Fetch the N_VAL selected val2017 images into DL/val2017/."""
    out_dir = DL / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in out_dir.glob("*.jpg")}
    todo = [im for im in val_imgs if im["file_name"] not in have]
    if not todo:
        return
    if use_official:
        for im in todo:
            curl(OFFICIAL_IMG.format(split="val2017", fname=im["file_name"]),
                 out_dir / im["file_name"], max_time=120)
    else:
        zpath = DL / "val2017.zip"
        curl(HF["val_zip"], zpath)
        wanted = {im["file_name"] for im in todo}
        print(f"  extracting {len(wanted)} val images from val2017.zip")
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                base = Path(name).name
                if base in wanted:
                    with z.open(name) as fsrc, open(out_dir / base, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    wanted.discard(base)
        assert not wanted, f"missing in mirror zip: {sorted(wanted)[:5]}"


def get_train_images(calib_imgs: list[dict], use_official: bool) -> None:
    """Fetch the N_CALIB selected train2017 images into DL/train2017/."""
    out_dir = DL / "train2017"
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in out_dir.glob("*.jpg")}
    todo = [im for im in calib_imgs if im["file_name"] not in have]
    if not todo:
        return
    if use_official:
        for im in todo:
            curl(OFFICIAL_IMG.format(split="train2017", fname=im["file_name"]),
                 out_dir / im["file_name"], max_time=120)
        return
    from remote_zip_fetch import fetch_files
    print(f"  remote-reading {len(todo)} train2017 images from the 19 GB "
          f"mirror zip (HTTP ranges only)")
    tmp = out_dir / "_remote"
    tmp.mkdir(exist_ok=True)
    wanted = [im["file_name"] for im in todo]
    try:
        written = fetch_files(HF["train_zip"], wanted, tmp)
    except Exception as e:
        raise RuntimeError(f"remote zip extraction failed: {e}")
    for fn in written:
        shutil.move(tmp / fn, out_dir / fn)
    tmp.rmdir()


def verify_images(imgs: list[dict], directory: Path) -> None:
    """Integrity: file_name + width + height must match official annotations."""
    from PIL import Image
    for im in imgs:
        p = directory / im["file_name"]
        assert p.exists(), f"{p} missing"
        with Image.open(p) as w:
            assert (w.width, w.height) == (im["width"], im["height"]), \
                f"dims mismatch for {im['file_name']}: got {(w.width, w.height)} " \
                f"want {(im['width'], im['height'])}"


def coco_to_yolo_labels(ann_json: dict, wanted_ids: set[int]) -> dict[int, list[str]]:
    """COCO absolute xywh boxes -> normalized YOLO txt lines (cls cx cy w h)."""
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(
        sorted(ann_json["categories"], key=lambda c: c["id"]))}
    dims = {im["id"]: (im["width"], im["height"]) for im in ann_json["images"]}
    per_img: dict[int, list[str]] = {i: [] for i in wanted_ids}
    for a in ann_json["annotations"]:
        if a["image_id"] not in per_img or a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        W, H = dims[a["image_id"]]
        cx, cy = x + w / 2, y + h / 2
        per_img[a["image_id"]].append(
            f"{cat_id_to_yolo[a['category_id']]} {cx / W:.6f} {cy / H:.6f} "
            f"{w / W:.6f} {h / H:.6f}")
    return per_img


def main() -> None:
    DATA.mkdir(exist_ok=True)
    use_official = official_host_reachable()
    print(f"[1/5] COCO annotations (official host: {use_official})")
    val_ann, train_ann = load_annotations(use_official)

    print(f"[2/5] deterministic subsets (seed={SEED}): "
          f"{N_VAL} val2017 + {N_CALIB} train2017 (disjoint by split)")
    val_imgs = pick(val_ann["images"], N_VAL)
    calib_imgs = pick(train_ann["images"], N_CALIB)
    assert not ({i['id'] for i in val_imgs} & {i['id'] for i in calib_imgs})

    print("[3/5] fetching images")
    get_val_images(val_imgs, use_official)
    get_train_images(calib_imgs, use_official)
    print("  verifying against official annotation metadata")
    verify_images(val_imgs, DL / "val2017")
    verify_images(calib_imgs, DL / "train2017")

    print("[4/5] building YOLO-format val subset + calibration list")
    vimg = DATA / "val_subset" / "images"
    vlbl = DATA / "val_subset" / "labels"
    for d in (vimg, vlbl):
        d.mkdir(parents=True, exist_ok=True)
    labels = coco_to_yolo_labels(val_ann, {im["id"] for im in val_imgs})
    n_boxes = 0
    for im in val_imgs:
        shutil.copyfile(DL / "val2017" / im["file_name"], vimg / im["file_name"])
        stem = Path(im["file_name"]).stem
        lines = labels[im["id"]]
        n_boxes += len(lines)
        (vlbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    cimg = DATA / "calib" / "images"
    cimg.mkdir(parents=True, exist_ok=True)
    calib_list = []
    for im in calib_imgs:
        shutil.copyfile(DL / "train2017" / im["file_name"], cimg / im["file_name"])
        calib_list.append(str((cimg / im["file_name"]).relative_to(REPO)))

    names = {i: c["name"] for i, c in enumerate(
        sorted(val_ann["categories"], key=lambda c: c["id"]))}
    (DATA / "val_subset.yaml").write_text(
        f"path: {(DATA / 'val_subset').resolve()}\ntrain: images\n"
        f"val: images\nnames: {names}\n")
    (DATA / "calib_list.txt").write_text("\n".join(calib_list) + "\n")

    print("[5/5] writing manifest (ids, sha256, licenses, sources)")
    lic = {l["id"]: l for l in val_ann["licenses"]}

    def lic_of(im):
        l = lic.get(im.get("license_id"),
                    {"name": "COCO/Flickr (see cocodataset.org terms)",
                     "url": "https://cocodataset.org/#termsofuse"})
        return l["name"], l["url"]
    manifest = {
        "note": "Educational COCO Subset - Not Official COCO Benchmark. "
                "Reproducible selection: random.Random(42).sample(sorted-by-id images, N).",
        "seed": SEED,
        "annotation_source": "official instances_{val,train}2017.json "
                             "(images.cocodataset.org or byte-exact HF mirror)",
        "num_val": len(val_imgs), "num_calib": len(calib_imgs),
        "overlap": "none (val2017 vs train2017 splits)",
        "val_source": VAL_SOURCE, "calib_source": TRAIN_SOURCE,
        "val_boxes_total": n_boxes,
        "val_images": [{
            "id": im["id"], "file_name": im["file_name"],
            "sha256": sha256(vimg / im["file_name"]),
            "coco_url": im.get("coco_url", ""),
            "license": lic_of(im)[0],
            "license_url": lic_of(im)[1],
            "verified_dims_match_annotations": True,
        } for im in val_imgs],
        "calib_images": [{
            "id": im["id"], "file_name": im["file_name"],
            "sha256": sha256(cimg / im["file_name"]),
            "coco_url": im.get("coco_url", ""),
            "verified_dims_match_annotations": True,
        } for im in calib_imgs],
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(val_imgs)} val images ({n_boxes} boxes) + "
          f"{len(calib_imgs)} calibration images -> data/manifest.json")


if __name__ == "__main__":
    main()
