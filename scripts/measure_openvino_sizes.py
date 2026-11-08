#!/usr/bin/env python3
"""Measure actual OpenVINO XML/BIN artifacts from disk. No synthetic sizes.

This tool is for a local machine with exported models available; an AMD GPU is
not required just to measure file sizes. It never creates a model or performs
any inference. Existing v2.0.1 campaign hashes/data are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OFFICIAL_SHA256 = "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(name: str, directory: Path) -> dict:
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} IR directory is missing: {directory}")
    result = {"directory": str(directory.resolve()), "files": {}, "total_bytes": 0}
    for suffix in ("xml", "bin"):
        path = directory / f"yolo26l.{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name} IR must contain nonempty {path}")
        size = path.stat().st_size
        result["files"][path.name] = {"bytes": size, "sha256": sha256(path)}
        result["total_bytes"] += size
    result["size_mb_decimal"] = round(result["total_bytes"] / 1e6, 6)
    result["size_mib"] = round(result["total_bytes"] / (1024 ** 2), 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-dir", default="weights/yolo26l_openvino_model")
    parser.add_argument("--int8-dir", required=True, help="Exact identity-keyed INT8 export directory; never guess a cache")
    parser.add_argument("--weights", default="weights/yolo26l.pt")
    parser.add_argument("--output", default="validation/v2.0.2/MEASURED_MODEL_SIZES.json")
    args = parser.parse_args()

    def repo_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else REPO / candidate

    weights = repo_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Official weight file is missing: {weights}")
    model_digest = sha256(weights)
    if model_digest != OFFICIAL_SHA256:
        raise RuntimeError("Model SHA256 does not match the official frozen YOLO26l weight")
    fp32 = measure("FP32", repo_path(args.fp32_dir))
    int8 = measure("INT8", repo_path(args.int8_dir))
    key_file = repo_path(args.int8_dir) / "CACHE_KEY.json"
    if not key_file.is_file():
        raise FileNotFoundError(f"INT8 identity file missing: {key_file}")
    cache_identity = json.loads(key_file.read_text())
    if not isinstance(cache_identity, dict):
        raise ValueError("CACHE_KEY.json must be a JSON object")

    output = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_type": "actual XML/BIN file sizes; no GPU inference",
        "freshly_measured": True,
        "model_sha256": model_digest,
        "fp32": fp32,
        "int8": int8,
        "int8_identity_file": str(key_file.resolve()),
        "int8_identity_sha256": sha256(key_file),
        "int8_over_fp32_size_ratio": round(int8["total_bytes"] / fp32["total_bytes"], 8),
        "fp32_over_int8_size_ratio": round(fp32["total_bytes"] / int8["total_bytes"], 8),
        "notes": "Measured from actual on-disk bytes. Must not be reported as W7900 performance evidence.",
    }
    target = repo_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"FP32: {fp32['size_mb_decimal']} MB; INT8: {int8['size_mb_decimal']} MB; "
          f"size reduction: {output['fp32_over_int8_size_ratio']}x")
    print(f"Evidence: {target} ({sha256(target)})")


if __name__ == "__main__":
    main()
