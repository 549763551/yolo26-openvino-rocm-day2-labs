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

## 实测结果（v2.0.1 · 2026-11-08）

本分支最新数据来自 **[v2.0.1 真机 Campaign](validation/v2.0.1/campaigns/c20261108_1108/CAMPAIGN.json)**，3 轮独立测试、每组 50 次正式计时，统计采用各轮 P50 的中位数。旧的 `results/latest.json` 保存的是 **v2.0.0 历史数据，不是新版展示数据**。

| Lab | 优化对比 | v2.0.1 实测 | 代价与边界 |
|---|---|---:|---|
| A | EPYC CPU → W7900 ROCm GPU | **14.14×**（P50 191.28→13.53ms） | 硬件平台对比，单张端到端 |
| B | PyTorch CPU → OpenVINO CPU | **2.28×**（198.54→87.22ms） | 同一 AMD CPU，mAP −0.0049 |
| C | OpenVINO FP32 → INT8 | **1.26×**（98.30→78.14ms） | **mAP −0.0667**，精度不能忽略 |
| D-A | imgsz 640 → 512 | **约 1.3%** P50 降低（13.68→13.50ms） | mAP −0.0239；P95 反而变差 |
| D-B | Forward-only Batch 1 → 8 | **1.52×** 吞吐（95.2→144.9 img/s） | Batch 完成时间≠单请求延迟 |
| E | CPU+GPU 串行 → 异步流水线 | **1.42×** 吞吐（58.62→83.13 img/s） | 突发请求含排队，P50≈611ms |

**注意**：机器为 2× AMD EPYC 9334 + W7900D（gfx1100），但容器 CPU 配额只有 16 核。精度评估使用 120 张图片的教学 COCO 子集，不是官方完整 COCO Benchmark。本轮 Resolution/Batch 未记录可信显存数据，不得将缺失视为 0 GB。模型导出文件大小（≈100/26 MB）如显示，须明确为 v2.0.0 的历史参考值。

- [完整 v2.0.1 结果表](results/RESULTS_TABLE.md)
- [Slide 28–51 数据与教学映射](docs/SLIDE_28_51_EVIDENCE_MAP.md)
- [独立审计：53 次运行记录、96 个文件哈希](validation/v2.0.1/campaigns/c20261108_1108/AUDIT_REPORT.md)

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
