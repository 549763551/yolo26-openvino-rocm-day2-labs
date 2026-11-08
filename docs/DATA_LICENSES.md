# Data Licenses / 数据许可

## COCO images (educational subsets)

- Source: COCO 2017 dataset, downloaded from the official host
  `images.cocodataset.org` (annotations: `annotations_trainval2017.zip`;
  images: individual `val2017/*.jpg` / `train2017/*.jpg` files).
- COCO images come from Flickr under the licenses listed in the annotation
  files (Creative Commons Attribution 4.0 / 2.0 etc., per-image `license`
  field). The per-image license name and URL for every image we ship or
  reference is recorded in `data/manifest.json` (written by
  `scripts/prepare_data.py`).
- Images are **not committed to this repository**; they are downloaded at
  setup time for educational use. The manifest (ids, URLs, hashes) is
  committed as provenance.
- This is an **Educational COCO Subset — Not Official COCO Benchmark**.
  mAP numbers in this course are for internal comparisons only.

## Official YOLO26l weights

- Downloaded at setup time from the official Ultralytics release assets
  (`github.com/ultralytics/assets`). Never committed to this repository.
  SHA256 recorded in `weights/model_identity.json`.
- The Ultralytics package is AGPL-3.0; using their public pretrained weights
  via the official downloader for education/research is the intended usage.

## Software

- PyTorch ROCm wheels: AMD (repo.amd.com), BSD-style license.
- OpenVINO / NNCF: Apache-2.0 (Intel).
- Ultralytics: AGPL-3.0.
- This repository's own code: MIT (see LICENSE).
