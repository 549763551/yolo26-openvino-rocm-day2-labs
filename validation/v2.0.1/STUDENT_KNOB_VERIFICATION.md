# Student-Knob Verification (R3.9)

Every student-editable parameter was CHANGED and RE-RUN on the v2.0.1 code this round; each verifiably alters the measured output (no fake knobs, no pre-generated charts).

| Knob | Change tested | Result | Wall (s) |
|---|---|---|---|
| `STUDENT_CALIB_N` | 150→30 (separate INT8 identity) | CHANGED AS EXPECTED | 19.1 |
| `STUDENT_IMGSZ_B` | 512→416 (new latency measurement) | CHANGED AS EXPECTED | 4.2 |
| `STUDENT_QUEUE_DEPTH` | 4→1 (pipeline throughput) | CHANGED AS EXPECTED | 2.5 |
| `STUDENT_DEVICE` | cpu/cuda (device latency) | CHANGED AS EXPECTED | 2.1 |
| `STUDENT_HINT` | LATENCY/THROUGHPUT (real ov.Core config) | CHANGED AS EXPECTED | 2.9 |

**ALL 5 KNOBS VERIFIED**
