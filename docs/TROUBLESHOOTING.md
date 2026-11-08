# Troubleshooting / 排错指南

## Quick diagnosis / 快速诊断

```bash
bash scripts/doctor.sh
```

It checks: AMD CPU, gfx1100 GPU, ROCm torch pins, real GPU compute
(matmul + NMS), OpenVINO CPU plugin, official weights identity, data subsets.

## Common issues / 常见问题

### `torch.cuda.is_available()` is False

- Check `rocminfo | grep gfx` shows `gfx1100` and `/dev/kfd` exists.
- Confirm the venv has the AMD wheel:
  `.venv/bin/python -c "import torch; print(torch.__version__)"` must print
  `2.12.0+rocm7.14.1`. If it prints a PyPI version, some later install
  overwrote it — fix with
  `bash scripts/restore_env.sh` (re-runs the pinned ROCm install).
- Never "fix" this by installing PyPI torch — the course requires the AMD ROCm
  wheels.

### GPU compute fails or hangs

- `rocm-smi` to check the GPU is healthy; re-run the smoke test:
  `.venv/bin/python course/setup_checks.py`.
- Multi-user contention on /dev/kfd can cause slow first kernels — the
  warm-up phase exists precisely for this.

### `yolo26l.pt` download timed out

- The file is ~50 MB from the Ultralytics release assets; on slow links use
  `bash scripts/setup.sh` again (curl with retries, resumable) — it will skip
  everything already completed.

### COCO data download failed

- Re-run `bash scripts/setup.sh` (step 7 checks for `data/manifest.json`).
  Sources: `images.cocodataset.org` (annotations zip + individual images).

### OpenVINO export / NNCF quantization failed

- Ensure disk space (~2 GB free in the repo) and that
  `weights/yolo26l_openvino_model/yolo26l.xml` was fully written.
- Delete the partial directory and re-run the export cell; export is
  deterministic from the same weights.

### INT8 is not faster (or slower) on my CPU

- That is a *result*, not an error. This machine's EPYC 9334 (Zen 4) has
  AVX-512 VNNI but no AMX; oneDNN INT8 kernels may or may not beat FP32.
  Check `results/latest.json` labC stats and discuss the bottleneck.

### Jupyter kernel not found

- Re-register:
  `.venv/bin/python -m ipykernel install --user --name yolo26-day2
  --display-name "Python 3.12 (YOLO26 Day 02 · ROCm)"`.

### Restoring on a fresh Radeon Cloud instance

```bash
cd /workspace/yolo26-openvino-rocm-day2-labs && bash scripts/restore_env.sh
```

`/workspace` persists `.venv`, `weights/`, `data/`, `results/`; the script
re-runs health gates and only reinstalls what is missing.
