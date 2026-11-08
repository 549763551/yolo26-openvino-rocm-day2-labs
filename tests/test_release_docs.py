"""Prevent misleading release documentation and ungrounded tail-latency claims."""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO / "validation/v2.0.1/campaigns/c20261108_1108"
CAMPAIGN = json.loads((CAMPAIGN_DIR / "CAMPAIGN.json").read_text())


def p95(samples: list[float]) -> float:
    values = sorted(samples)
    k = 0.95 * (len(values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def measured_p95(key: str) -> float:
    per_round = []
    for round_record in CAMPAIGN["aggregate"][key]["rounds"]:
        run_id = round_record["run_id"]
        paths = list(CAMPAIGN_DIR.glob(f"*/{run_id}.csv"))
        assert len(paths) == 1, f"expected one real CSV for {run_id}, got {paths}"
        with paths[0].open() as handle:
            data = [float(row["ms"]) for row in csv.DictReader(handle)]
        assert len(data) == round_record["n"], run_id
        value = p95(data)
        assert abs(value - round_record["p95_ms"]) < 0.005, run_id
        per_round.append(value)
    from_csv = statistics.median(per_round)
    assert abs(from_csv - CAMPAIGN["aggregate"][key]["p95_median_ms"]) < 0.005
    return from_csv


def find_row_column(path: str, row_start: str, column_index: int) -> str:
    lines = (REPO / path).read_text().splitlines()
    matches = [line for line in lines if line.startswith(row_start)]
    assert len(matches) == 1, f"{path}: expected unique {row_start!r}, got {len(matches)}"
    cells = [cell.strip() for cell in matches[0].split("|")]
    return cells[column_index]


def test_release_p95_matches_raw_csv():
    pairs = [
        ("labA_cpu", "| AMD EPYC 9334 (PyTorch,", 3),
        ("labA_gpu", "| W7900D (PyTorch ROCm)", 3),
        ("labB_pt", "| PyTorch CPU ", 3),
        ("labB_ov", "| OpenVINO CPU ", 3),
    ]
    for doc in ("README.md", "README.en.md"):
        for key, prefix, col in pairs:
            expected = measured_p95(key)
            displayed = float(find_row_column(doc, prefix, col))
            assert abs(displayed - expected) <= 0.051, (doc, key, displayed, expected)
    result_keys = {
        "labA_cpu": "| Lab A · AMD CPU",
        "labA_gpu": "| Lab A · W7900D",
        "labB_pt": "| Lab B · AMD CPU / PyTorch",
        "labB_ov": "| Lab B · AMD CPU / OpenVINO",
        "labC_fp32": "| Lab C · OpenVINO FP32",
        "labC_int8": "| Lab C · OpenVINO INT8",
        "labD_res_640": "| Lab D-A · W7900D, imgsz 640",
        "labD_res_512": "| Lab D-A · W7900D, imgsz 512",
        "labD_b1": "| Lab D-B · W7900D, batch=1 forward",
        "labD_b8": "| Lab D-B · W7900D, batch=8 forward",
        "labD_b1_e2e": "| Lab D-B · W7900D, batch=1 E2E",
        "labD_b8_e2e": "| Lab D-B · W7900D, batch=8 E2E",
        "labE_serial": "| Lab E · serial",
        "labE_pipeline": "| Lab E · async pipeline",
    }
    for key, prefix in result_keys.items():
        expected = measured_p95(key)
        displayed = float(find_row_column("results/RESULTS_TABLE.md", prefix, 3).replace("**", ""))
        assert abs(displayed - expected) <= 0.006, (key, displayed, expected)


def test_no_unresolved_merge_conflict_markers():
    for path in [*REPO.glob("*.md"), *REPO.glob("docs/**/*.md"),
                 REPO / "results/RESULTS_TABLE.md",
                 REPO / "validation/v2.0.1/FINAL_REPORT.md"]:
        source = path.read_text()
        # literal regex with character classes avoids embedding active conflict markers
        assert not re.search(r"(?m)^(?:<{7}|={7}|>{7})(?![=])", source), str(path)


def test_exactly_24_slides_with_approved_topics():
    source = (REPO / "docs/SLIDE_28_51_EVIDENCE_MAP.md").read_text()
    slide_rows = re.findall(r"(?m)^\| (\d{2}) \| ([^|]+) \|", source)
    assert [int(i) for i, _ in slide_rows] == list(range(28, 52))
    required = {
        28: "模型训练完成", 29: "一张图片经过哪些步骤", 30: "推理时间到底花在哪里",
        31: "Latency", 32: "公平", 33: "Lab A", 34: "模型训练格式",
        35: "OpenVINO", 36: "YOLO26", 37: "OpenVINO",
        38: "Lab B", 39: "FP32", 40: "INT8", 41: "模型压缩",
        42: "Lab C", 43: "GPU", 44: "ROCm", 45: "输入分辨率",
        46: "Batch", 47: "GPU", 48: "Lab D", 49: "AMD CPU",
        50: "Lab E", 51: "优化",
    }
    for num, title in slide_rows:
        assert required[int(num)] in title, (num, title)


def test_provenance_and_no_missing_cold_start_log_reference():
    report = (REPO / "validation/v2.0.1/FINAL_REPORT.md").read_text()
    assert "COLDSTART_LOG.md" not in report
    assert (REPO / "validation/v2.0.1/INDEPENDENT_AUDIT_LOG.md").is_file()
    for path in ("README.md", "README.en.md", "docs/INSTRUCTOR_GUIDE.md",
                 "results/RESULTS_TABLE.md", "docs/SLIDE_28_51_EVIDENCE_MAP.md"):
        content = (REPO / path).read_text()
        assert "v2.0.0" in content and (
            "historical" in content.lower() or "历史" in content
        ), f"{path}: v2.0.0 model-size provenance not disclosed"


def test_tail_latency_in_teacher_notes_and_final_report():
    instructor = (REPO / "docs/INSTRUCTOR_GUIDE.md").read_text()
    final = (REPO / "validation/v2.0.1/FINAL_REPORT.md").read_text()
    assert "P95 470.6" in instructor and "P95 15.7" in instructor
    assert "470.6→15.7" in final
