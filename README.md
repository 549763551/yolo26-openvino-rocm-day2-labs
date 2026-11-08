# YOLO26 Day 02 — AI 推理加速：OpenVINO × AMD ROCm / AI Inference Acceleration with OpenVINO & AMD ROCm

**中文全称**：从模型到系统：AI 推理加速与 AMD CPU/GPU 异构协同
**English**: From Model to System — AI Inference Acceleration and Heterogeneous CPU+GPU Computing on AMD

面向大四学生的推理加速实战课程：**一块官方 YOLO26l 权重，两种 AMD 计算
设备（EPYC CPU + Radeon PRO W7900），六份双语 Notebook，六组真实性能
实验，全部证据可独立核验。**

A hands-on senior-undergraduate course: **one official YOLO26l model, two AMD
compute devices (EPYC CPU + Radeon PRO W7900 48 GB), six bilingual notebooks,
six real measured experiments — with independently auditable evidence.** No
fabricated numbers anywhere: every figure traces to immutable per-iteration
CSV samples covered by SHA256SUMS and an independent audit script.

---

## 硬件与软件 / Machine & software (v2.0.1 measurement machine)

| | |
|---|---|
| CPU | 2× AMD EPYC 9334 (Zen 4, 64C/128T, 2 NUMA nodes, AVX-512 + VNNI, **无 Intel AMX**)；容器 CPU 配额 16 核（与物理拓扑分别记录） |
| GPU | AMD Radeon PRO W7900D 48 GB (gfx1100, RDNA3)，共享宿主机（负载已记录） |
| PyTorch | `2.12.0+rocm7.14.1` (AMD ROCm wheel, `repo.amd.com`，版本锁定) |
| OpenVINO | 2026.4.1 CPU plugin (oneDNN) — OpenVINO 对 AMD CPU 的官方验证范围有限，本仓库以真机实测为准 |
| Ultralytics | 官方 `yolo26l.pt`（80 类 COCO 检测，sha256 `9fe3c544…db26`，见 `weights/model_identity.json`） |

## 六份 Notebook / The six labs

| # | Notebook | 核心问题 / Question |
|---|---|---|
| 00 | [AMD 环境与官方 YOLO26l](notebooks/00_AMD_Environment_and_YOLO26l.ipynb) | 模型到底跑在哪个设备上？/ Which device & runtime path is it really running on? |
| A | [Lab A: CPU vs GPU 基线](notebooks/01_Lab_A_CPU_vs_GPU_Baseline.ipynb) | 同一模型，CPU 与 GPU 谁快？Amdahl 定律的固定开销在哪？/ Same model — which device wins? Where is the fixed overhead? |
| B | [Lab B: OpenVINO 加速 AMD CPU](notebooks/02_Lab_B_OpenVINO_AMD_CPU.ipynb) | 不换 CPU 只换运行时，会怎样？/ Same CPU, different runtime? |
| C | [Lab C: INT8 量化](notebooks/03_Lab_C_OpenVINO_INT8.ipynb) | 更少的位数 = 更快的计算？代价多大？/ Fewer bits, faster math? At what cost? |
| D | [Lab D: ROCm GPU 优化](notebooks/04_Lab_D_ROCm_GPU_Optimization.ipynb) | 分辨率与 Batch 如何影响 GPU？/ Resolution & batch effects |
| E | [Lab E: CPU+GPU 异步流水线](notebooks/05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb) | 异构协同能更快吗？重叠如何证明？/ Can CPU+GPU overlap beat serial — and how is overlap proven? |

每份 Notebook 统一模板（中英双语）：**本节目标 → 核心原理 → 先预测 →
运行基线 → 只改一个因素 → 对比真实数据 → 为什么 → 学生实践 → 常见错误
→ 一句话总结**。

## 快速开始 / Quick start

```bash
git clone https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs.git
cd yolo26-openvino-rocm-day2-labs
bash scripts/setup.sh                        # 一键安装（硬件门禁 + ROCm torch + 数据校验/修复 + 自检）
bash scripts/doctor.sh                       # 环境诊断（只读）
bash scripts/run_all.sh --profile quick      # 课堂快速实测
bash scripts/run_all.sh --profile full       # 正式 Benchmark
source .venv/bin/activate && jupyter lab     # 打开六份 Notebook
```

Notebook 执行门禁（真实运行全部 6 份，而非仅检查结构）：

```bash
bash scripts/validate_notebooks.sh             # 结构校验（CI 使用）
bash scripts/validate_notebooks.sh --execute   # 真实执行 6/6 + 输出核验
```

证据核验 / Evidence verification（任何人可独立复核）：

```bash
python scripts/verify_evidence.py --campaign validation/v2.0.1
python scripts/audit_benchmarks.py --campaign validation/v2.0.1/campaigns/<id>
```

## 实测结果（v2.0.1 真机复测）/ Measured results (v2.0.1 re-measured)

**Campaign `c20261108_1108`**（2026-11-08 · 3 轮独立测量 × 每变体 50 个
正式样本 · warm-up 10 已排除 · 多轮取中位 · 模型 sha256 `9fe3c544…db26` ·
AMD EPYC 9334 + W7900D · 宿主机共享、负载与限流计数已记录）。完整证据：
`validation/v2.0.1/campaigns/c20261108_1108/`（CSV + summary + SHA256SUMS +
独立审计 `AUDIT_REPORT.md`）。

v2.0.0 的历史数据仍保留在 `results/raw/`（仅作历史参考，不再用于对外结论）。

### 1 · Lab A — CPU vs GPU（端到端，同一官方模型，batch 1）

| 方案 | P50 ms | P95 ms | 吞吐 img/s |
|---|---|---|---|
| AMD EPYC 9334 (PyTorch, 8 线程, 16-CPU 亲和) | 191.3 | 470.6 | 5.2 |
| W7900D (PyTorch ROCm) | 13.5 | 15.7 | 73.9 |

**GPU 为 CPU 的 14.14×**（平台对比，非软件优化；轮间波动 CPU 7.5% / GPU 5.8%）

### 2 · Lab B — 同一 CPU，两种运行时

| Runtime | P50 ms | P95 ms | mAP50-95 |
|---|---|---|---|
| PyTorch CPU | 198.5 | 441.0 | 0.6398 |
| OpenVINO CPU | 87.2 | 93.8 | 0.6349 |

**OpenVINO 为 PyTorch 的 2.28×** · ΔmAP −0.0049（精度一致）· 首次编译 ~1–2 s 一次性

### 3 · Lab C — INT8 量化（NNCF MIXED，150 张真实 train2017 校准，本轮重新量化）

| 精度 | 大小 MB | P50 ms | mAP50-95 |
|---|---|---|---|
| FP32 | 历史约100¹ | 98.3 | 0.6349 |
| INT8 | 历史约26¹ | 78.1 | 0.5682 |

**1.26× 提速（v2.0.1 真机）· ΔmAP −0.0667（明显的精度代价，绝非无损）**

¹ 模型大小约 100/26 MB 与约 3.8× 缩小来自 **v2.0.0 的历史真实导出**，不是 v2.0.1 重新测得。新工件文件大小需要独立实际测量；不应将这两个历史数值作为本次新测结论。

### 4 · Lab D-A — 分辨率（W7900，端到端 batch 1）

| imgsz | P50 ms | mAP50-95 |
|---|---|---|
| 640 | 13.68 | 0.6398 |
| 512 | 13.50 | 0.6159 |

端到端仅 **−1.3%**（batch 1 时固定前后处理主导端到端；纯前向部分随分辨率
下降更多）· ΔmAP −0.024（速度-精度权衡真实存在）

### 5 · Lab D-B — Batch（W7900，imgsz 640）

| Batch | 前向吞吐 img/s | 口径说明 |
|---|---|---|
| 1 | 95.2 | 前向 10.5 ms/req；端到端 13.6 ms/req |
| 8 | 144.9 | 前向 55.2 ms/批；端到端 80.0 ms/8 张请求 |

**前向吞吐 1.52×** · 55.2÷8=6.9 ms 只是单张等效计算成本，**不是任何用户的
请求延迟**（两种口径均已分别测量并标注）

### 6 · Lab E — CPU+GPU 异步流水线（100 张突发）

| 模式 | 总吞吐 img/s | 单张 e2e P50（口径） |
|---|---|---|
| 串行 | 58.6 | 14.0 ms（无队列，纯执行） |
| 异步流水线 | 83.1 | 603.5 ms（三轮 P50 的中位数；突发提交→出结果，含排队） |

**吞吐 1.42×** · 逐目标一致性 **100/100 张**（类别/坐标/置信度逐对比对，
min IoU = 1.0，max Δconf = 0.0 —— 位级一致）· 两种延迟口径分开报告，不可
相减；重叠由实测阶段时间轴证明（Notebook 05 §5b）

*数据来源：`validation/v2.0.1/campaigns/c20261108_1108/`（每轮 Run ID +
逐次延迟 CSV + SHA256SUMS + `CAMPAIGN.json` + 独立审计报告）。图表位于
`assets/slides/`，由该 campaign 重新生成。*

## 教学主线 / Teaching arc

1. **少计算** — 输入尺寸（Lab D-A）、INT8 量化（Lab C）
2. **快计算** — OpenVINO AMD CPU（Lab B）、ROCm W7900（Lab A、D）
3. **并行计算** — Batch（Lab D-B）、异步流水线与 CPU/GPU 协同（Lab E）

课程最终目标：让学生能回答 **"为什么快了、快了多少、代价是什么、如何证明"**。

## 诚实规则 / Honesty rules

- 只使用官方 `yolo26l.pt`（禁止换小模型刷数据 / never a smaller model）。
- 允许实验结果为"没有加速"或"更慢"，并解释原因（we report what we measure）。
- 量化/分辨率/Batch 实验必须同时报告精度或资源代价（costs are always reported）。
- COCO 数据为教学子集（*Educational COCO Subset — Not Official COCO
  Benchmark*），清单与 SHA256 见 `data/manifest.json`。
- 性能测量在共享宿主机上进行：每轮记录系统负载与 cgroup 限流计数并随
  证据存档（CPU 基准亲和到 16 个安静 CPU，与 16 核配额匹配）。

## 文档 / Docs

- [学生指南 Student Guide](docs/STUDENT_GUIDE.md)
- [教师指南 Instructor Guide](docs/INSTRUCTOR_GUIDE.md)
- [测量协议 Benchmark Protocol](docs/BENCHMARK_PROTOCOL.md)
- [排错 Troubleshooting](docs/TROUBLESHOOTING.md)
- [Slide 28–51 证据映射 Evidence Map](docs/SLIDE_28_51_EVIDENCE_MAP.md)
- [数据许可 Data Licenses](docs/DATA_LICENSES.md)
- v2.0.1 验证：`validation/v2.0.1/`（基线审计 / 诊断报告 / campaign 证据 /
  最终报告 FINAL_REPORT.md）

## License

MIT (code & docs). Weights are downloaded at setup time from Ultralytics
(AGPL-3.0 package); COCO images are downloaded at setup time under their
original Flickr licenses (see `docs/DATA_LICENSES.md`). Neither is committed.
