# YOLO26 Day 02 — AI 推理加速：OpenVINO × AMD ROCm / AI Inference Acceleration with OpenVINO & AMD ROCm

**中文全称**：从模型到系统：AI 推理加速与 AMD CPU/GPU 异构协同
**English**: From Model to System — AI Inference Acceleration and Heterogeneous CPU+GPU Computing on AMD

面向大四学生的推理加速实战课程：**一块官方 YOLO26l 权重，两种 AMD 计算设备
（EPYC CPU + Radeon PRO W7900），六份双语 Notebook，六组真实性能实验。**

A hands-on senior-undergraduate course: **one official YOLO26l model, two AMD
compute devices (EPYC CPU + Radeon PRO W7900 48 GB), six bilingual notebooks,
six real measured experiments.** No fabricated numbers anywhere — every figure
in this repository is generated from raw per-iteration CSV samples you can
audit (`results/raw/`).

---

## 硬件与软件 / Machine & software

| | |
|---|---|
| CPU | 2× AMD EPYC 9334 (Zen 4, 64C/128T, AVX-512, 无 AMX) |
| GPU | AMD Radeon PRO W7900 48 GB (gfx1100, RDNA3) |
| PyTorch | `2.12.0+rocm7.14.1` (AMD ROCm wheel, `repo.amd.com`) |
| OpenVINO | CPU plugin (oneDNN) — 注意：OpenVINO 对 AMD CPU 的官方验证范围有限，本仓库以真机实测为准 |
| Ultralytics | 官方 `yolo26l.pt`（80 类 COCO 检测，SHA256 见 `weights/model_identity.json`） |

## 六份 Notebook / The six labs

| # | Notebook | 核心问题 / Question |
|---|---|---|
| 00 | [AMD 环境与官方 YOLO26l](notebooks/00_AMD_Environment_and_YOLO26l.ipynb) | 模型到底跑在哪个设备上？/ Which device is it really running on? |
| A | [Lab A: CPU vs GPU 基线](notebooks/01_Lab_A_CPU_vs_GPU_Baseline.ipynb) | 同一模型，CPU 与 GPU 谁快？/ Same model — which device wins? |
| B | [Lab B: OpenVINO 加速 AMD CPU](notebooks/02_Lab_B_OpenVINO_AMD_CPU.ipynb) | 不换 CPU 只换运行时，会怎样？/ Same CPU, different runtime? |
| C | [Lab C: INT8 量化](notebooks/03_Lab_C_OpenVINO_INT8.ipynb) | 更少的位数 = 更快的计算？/ Fewer bits, faster math? |
| D | [Lab D: ROCm GPU 优化](notebooks/04_Lab_D_ROCm_GPU_Optimization.ipynb) | 分辨率与 Batch 如何影响 GPU？/ Resolution & batch effects |
| E | [Lab E: CPU+GPU 异步流水线](notebooks/05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb) | 异构协同能更快吗？/ Can CPU+GPU overlap beat serial? |

每份 Notebook 8–12 个核心 Cell，统一模板：**学什么 → 核心原理 → 基线 →
只改一个因素 → 对比真实数据 → 为什么 → 亲自修改参数 → 一句话总结**（中英双语）。

## 快速开始 / Quick start

```bash
git clone https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs.git
cd yolo26-openvino-rocm-day2-labs
bash scripts/setup.sh                     # 一键安装（ROCm torch + OpenVINO + 数据 + 自检）
bash scripts/run_all.sh --profile quick   # 课堂快速实测
bash scripts/run_all.sh --profile full    # 正式 Benchmark
source .venv/bin/activate && jupyter lab  # 打开六份 Notebook
```

- 环境诊断 `bash scripts/doctor.sh`
- 云主机恢复 `bash scripts/restore_env.sh`
- 测试 `python -m pytest -q`

## 实测结果（真机）/ Measured results (real machine)

### Six measured results / 六组实测结果

profile: **full** · generated: 2026-11-08T09:56:56 · model: official yolo26l.pt (80-class COCO detection)

All values measured on this machine (2× AMD EPYC 9334 + Radeon PRO W7900, ROCm PyTorch 2.12.0+rocm7.14.1). Raw samples: `results/raw/`.

| Lab | Device | P50 ms | P95 ms | Throughput img/s |
|---|---|---|---|---|
| A baseline | AMD CPU (PyTorch) | 124.8 | 160.0 | 8.0 |
| A baseline | W7900 (PyTorch ROCm) | 13.3 | 26.1 | 75.3 |

GPU vs CPU P50 speedup: **9.40x**

| Lab | Runtime (same AMD CPU) | P50 ms | P95 ms | mAP50-95 |
|---|---|---|---|---|
| B runtime | PyTorch | 154.6 | 183.6 | 0.6398 |
| B runtime | OpenVINO | 50.1 | 75.4 | 0.6349 |

OpenVINO vs PyTorch P50: **3.09x** · ΔmAP: -0.0049

| Lab | Precision | Size MB | P50 ms | mAP50-95 |
|---|---|---|---|---|
| C quant | FP32 | 100 | 49.5 | 0.6349 |
| C quant | INT8 | 26 | 28.2 | 0.5682 |

INT8 vs FP32 P50: **1.75x** · size 3.79x smaller · ΔmAP: -0.0667

| Lab | Config | P50 ms | mAP50-95 | VRAM GB |
|---|---|---|---|---|
| D-A resolution | imgsz 640 | 13.2 | 0.6398 | 0.36 |
| D-A resolution | imgsz 512 | 12.5 | 0.6159 | 0.33 |

640→512 latency: **1.06x faster** · ΔmAP: -0.0239

| Lab | Batch | Throughput img/s | Latency semantics |
|---|---|---|---|
| D-B batch | 1 | 97.9 | per-request 10.2 ms |
| D-B batch | 8 | 145.6 | batch 54.9 ms (equiv 6.9 ms/img, NOT request latency) |

Batch throughput gain: **1.49x** · VRAM peak 1.17 GB

| Lab | Mode | Total s | img/s | e2e P50 ms |
|---|---|---|---|---|
| E pipeline | serial | 1.4 | 72.3 | 13 |
| E pipeline | async CPU+GPU | 1.1 | 87.4 | 579 |

Pipeline throughput gain: **1.21x** · detection consistency: PASS (100 images)

*数据来源：`results/latest.json` + `results/raw/`（每次运行保存逐次延迟样本 CSV 与
SHA256）。图表位于 `assets/slides/`，全部由原始数据生成，可复现。*

## 教学主线 / Teaching arc

1. **少计算** — 输入尺寸（Lab D-A）、INT8 量化（Lab C）
2. **快计算** — OpenVINO AMD CPU（Lab B）、ROCm W7900（Lab A、D）
3. **并行计算** — Batch（Lab D-B）、异步流水线与 CPU/GPU 协同（Lab E）

课程最终目标：让学生能回答 **“为什么快了、快了多少、代价是什么、如何证明”**。

## 诚实规则 / Honesty rules

- 只使用官方 `yolo26l.pt`（禁止换小模型刷数据 / never a smaller model）。
- 允许实验结果为"没有加速"或"更慢"，并解释原因（we report what we measure）。
- 量化/分辨率/Batch 实验必须同时报告精度或资源代价（costs are always reported）。
- COCO 数据为教学子集（*Educational COCO Subset — Not Official COCO Benchmark*），
  清单与 SHA256 见 `data/manifest.json`。

## 文档 / Docs

- [学生指南 Student Guide](docs/STUDENT_GUIDE.md)
- [教师指南 Instructor Guide](docs/INSTRUCTOR_GUIDE.md)
- [测量协议 Benchmark Protocol](docs/BENCHMARK_PROTOCOL.md)
- [排错 Troubleshooting](docs/TROUBLESHOOTING.md)
- [Slide 28–51 证据映射 Evidence Map](docs/SLIDE_28_51_EVIDENCE_MAP.md)
- [数据许可 Data Licenses](docs/DATA_LICENSES.md)

## License

MIT (code & docs). Weights are downloaded at setup time from Ultralytics
(AGPL-3.0 package); COCO images are downloaded at setup time under their
original Flickr licenses (see `docs/DATA_LICENSES.md`). Neither is committed.
