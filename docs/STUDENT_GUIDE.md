# Student Guide / 学生指南

**Course / 课程**: YOLO26 Day 02 — From Model to System: AI Inference Acceleration
with OpenVINO & AMD ROCm / 从模型到系统：AI 推理加速与 AMD CPU/GPU 异构协同

## 0. What you need / 你需要什么

- A Linux machine with an **AMD CPU** and the **Radeon PRO W7900** (gfx1100) —
  this repository's lab machine. / 本课程实验机。
- Python 3.12 (installed automatically). / 自动安装。

## 1. One-time setup / 一次性安装

```bash
git clone https://github.com/AIwork4me/yolo26-openvino-rocm-day2-labs.git
cd yolo26-openvino-rocm-day2-labs
bash scripts/setup.sh
```

`setup.sh` is idempotent — safe to re-run. It installs the AMD ROCm PyTorch
wheels (`torch 2.12.0+rocm7.14.1`), OpenVINO, NNCF, Ultralytics, Jupyter,
downloads the **official `yolo26l.pt`** and a fixed COCO subset, then runs
real compute checks on the CPU and the W7900.

Check your environment any time / 随时检查环境:

```bash
bash scripts/doctor.sh
```

## 2. Run the results / 跑出结果

Quick, classroom profile (fewer repetitions, same real model):

```bash
bash scripts/run_all.sh --profile quick
```

Formal benchmark (more repetitions, citable numbers):

```bash
bash scripts/run_all.sh --profile full
```

Open the notebooks / 打开 Notebook:

```bash
source .venv/bin/activate
jupyter lab
```

Select the kernel **Python 3.12 (YOLO26 Day 02 · ROCm)**.

## 3. The six notebooks / 六份 Notebook

| # | File | Core question / 核心问题 | Time |
|---|------|--------------------------|------|
| 00 | `00_AMD_Environment_and_YOLO26l.ipynb` | What device is YOLO26l really running on? / 模型到底跑在哪个设备上？ | 10 min |
| A | `01_Lab_A_CPU_vs_GPU_Baseline.ipynb` | CPU or GPU — who is faster? / 谁更快？ | 15 min |
| B | `02_Lab_B_OpenVINO_AMD_CPU.ipynb` | Same CPU, different runtime — what changes? / 换运行时会怎样？ | 20 min |
| C | `03_Lab_C_OpenVINO_INT8.ipynb` | Can INT8 compute faster with less memory? / INT8 更快吗？ | 20–30 min |
| D | `04_Lab_D_ROCm_GPU_Optimization.ipynb` | Resolution & batch — how does GPU speed move? / 分辨率与 Batch 的影响 | 20 min |
| E | `05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb` | Can CPU+GPU working together beat serial? / 异步流水线更快吗？ | 20 min |

Every notebook follows the same 8-step template: learn → idea → baseline →
change ONE thing → compare real numbers → why → try it yourself → takeaway.
每份 Notebook 都遵循：学什么 → 核心原理 → 基线 → 只改一个因素 → 对比真实数据 →
为什么 → 亲自修改 → 一句话总结。

## 4. Hard rules of this course / 课程硬规则

1. **Only the official `yolo26l.pt`** (80-class COCO detection). Never swap in
   a smaller model for better numbers. / 只用官方 YOLO26l，绝不换小模型刷数字。
2. **Never fabricate numbers.** If an optimization does not help, report and
   explain that. / 绝不编造数据；没有加速就如实报告并解释。
3. **Always report the cost side**: accuracy, memory, latency semantics.
   / 永远同时报告代价：精度、内存、延迟口径。
4. The COCO subsets here are **educational subsets**, not official COCO
   benchmarks — mAP values are comparable *within* this course only.
   / 本课程 mAP 仅用于内部对比。

## 5. What to submit / 作业提交

- Your predictions vs measurements for each lab (each notebook has a
  "Try it yourself" cell). / 每个实验的预测与实测对比。
- One paragraph per lab answering: *why faster/slower, how much, at what cost,
  how do you know?* / 每个实验一段话：为什么快/慢、快多少、代价是什么、如何证明。

## 6. Troubleshooting / 排错

See `docs/TROUBLESHOOTING.md` and run `bash scripts/doctor.sh`.
