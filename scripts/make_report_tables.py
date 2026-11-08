#!/usr/bin/env python3
"""Generate the real-results markdown tables for README and FINAL_REPORT
from results/latest.json (no hand-typed numbers anywhere)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LATEST = REPO / "results" / "latest.json"


def x(a, b):
    return f"{a / b:.2f}x" if b else "n/a"


def main():
    d = json.loads(LATEST.read_text())
    L = []
    A = L.append
    A("### Six measured results / 六组实测结果\n")
    A(f"profile: **{d['profile']}** · generated: {d['generated']} · "
      "model: official yolo26l.pt (80-class COCO detection)\n")
    A("All values measured on this machine (2× AMD EPYC 9334 + Radeon PRO "
      "W7900, ROCm PyTorch 2.12.0+rocm7.14.1). Raw samples: `results/raw/`.\n")

    if "labA" in d:
        c, g = d["labA"]["cpu"]["stats"], d["labA"]["gpu"]["stats"]
        A("| Lab | Device | P50 ms | P95 ms | Throughput img/s |")
        A("|---|---|---|---|---|")
        A(f"| A baseline | AMD CPU (PyTorch) | {c['p50_ms']:.1f} | {c['p95_ms']:.1f} | {1000/c['p50_ms']:.1f} |")
        A(f"| A baseline | W7900 (PyTorch ROCm) | {g['p50_ms']:.1f} | {g['p95_ms']:.1f} | {1000/g['p50_ms']:.1f} |")
        A(f"\nGPU vs CPU P50 speedup: **{x(c['p50_ms'], g['p50_ms'])}**\n")

    if "labB" in d:
        p, o = d["labB"]["pytorch_cpu"]["stats"], d["labB"]["openvino_cpu"]["stats"]
        mp, mo = d["labB"]["map"]["pytorch"]["map50_95"], d["labB"]["map"]["openvino"]["map50_95"]
        A("| Lab | Runtime (same AMD CPU) | P50 ms | P95 ms | mAP50-95 |")
        A("|---|---|---|---|---|")
        A(f"| B runtime | PyTorch | {p['p50_ms']:.1f} | {p['p95_ms']:.1f} | {mp:.4f} |")
        A(f"| B runtime | OpenVINO | {o['p50_ms']:.1f} | {o['p95_ms']:.1f} | {mo:.4f} |")
        A(f"\nOpenVINO vs PyTorch P50: **{x(p['p50_ms'], o['p50_ms'])}** · "
          f"ΔmAP: {mo-mp:+.4f}\n")

    if "labC" in d:
        f5, i5 = d["labC"]["fp32"]["stats"], d["labC"]["int8"]["stats"]
        mf, mi = d["labC"]["map"]["fp32"]["map50_95"], d["labC"]["map"]["int8"]["map50_95"]
        A("| Lab | Precision | Size MB | P50 ms | mAP50-95 |")
        A("|---|---|---|---|---|")
        A(f"| C quant | FP32 | {d['labC']['size_mb']['fp32']:.0f} | {f5['p50_ms']:.1f} | {mf:.4f} |")
        A(f"| C quant | INT8 | {d['labC']['size_mb']['int8']:.0f} | {i5['p50_ms']:.1f} | {mi:.4f} |")
        A(f"\nINT8 vs FP32 P50: **{x(f5['p50_ms'], i5['p50_ms'])}** · "
          f"size {d['labC']['size_mb']['fp32']/d['labC']['size_mb']['int8']:.2f}x smaller · "
          f"ΔmAP: {mi-mf:+.4f}\n")

    if "labD" in d:
        r6, r5 = d["labD"]["resolution"]["640"], d["labD"]["resolution"]["512"]
        b1, b8 = d["labD"]["batch"]["1"], d["labD"]["batch"]["8"]
        A("| Lab | Config | P50 ms | mAP50-95 | VRAM GB |")
        A("|---|---|---|---|---|")
        A(f"| D-A resolution | imgsz 640 | {r6['stats']['p50_ms']:.1f} | {r6['map']['map50_95']:.4f} | {r6['vram_gb']:.2f} |")
        A(f"| D-A resolution | imgsz 512 | {r5['stats']['p50_ms']:.1f} | {r5['map']['map50_95']:.4f} | {r5['vram_gb']:.2f} |")
        A(f"\n640→512 latency: **{x(r6['stats']['p50_ms'], r5['stats']['p50_ms'])} faster** · "
          f"ΔmAP: {r5['map']['map50_95']-r6['map']['map50_95']:+.4f}\n")
        A("| Lab | Batch | Throughput img/s | Latency semantics |")
        A("|---|---|---|---|")
        A(f"| D-B batch | 1 | {b1['throughput_img_s']:.1f} | per-request {b1['p50_ms']:.1f} ms |")
        A(f"| D-B batch | 8 | {b8['throughput_img_s']:.1f} | batch {b8['p50_batch_ms']:.1f} ms (equiv {b8['per_image_equivalent_ms']:.1f} ms/img, NOT request latency) |")
        A(f"\nBatch throughput gain: **{x(b8['throughput_img_s'], b1['throughput_img_s'])}** · "
          f"VRAM peak {d['labD']['batch']['vram_gb']:.2f} GB\n")

    if "labE" in d:
        s, p, c = d["labE"]["serial"], d["labE"]["pipeline"], d["labE"]["compare"]
        A("| Lab | Mode | Total s | img/s | e2e P50 ms |")
        A("|---|---|---|---|---|")
        A(f"| E pipeline | serial | {s['total_ms']/1000:.1f} | {s['images_per_s']:.1f} | {s['e2e_p50_ms']:.0f} |")
        A(f"| E pipeline | async CPU+GPU | {p['total_ms']/1000:.1f} | {p['images_per_s']:.1f} | {p['e2e_p50_ms']:.0f} |")
        A(f"\nPipeline throughput gain: **{c['throughput_gain_x']:.2f}x** · "
          f"detection consistency: {'PASS' if c['consistent'] else 'MISMATCH'} "
          f"({c['files_compared']} images)\n")

    table = "\n".join(L)
    out = REPO / "results" / "RESULTS_TABLE.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(table + "\n")
    print(table)
    print(f"\n[written] {out.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
