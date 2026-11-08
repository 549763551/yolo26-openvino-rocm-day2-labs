# Independent Benchmark Audit

Campaign: `validation/v2.0.1/campaigns/c20261108_1108`
Model sha256: `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`
Files hash-verified: 96
Runs recomputed: 53

## Per-run recomputation

- labA_cpu_vs_gpu_pytorch_cpu_20261108_110915: n=50 p50=188.77 p95=470.63 OK
- labA_cpu_vs_gpu_pytorch_cpu_20261108_111137: n=50 p50=191.28 p95=438.91 OK
- labA_cpu_vs_gpu_pytorch_cpu_20261108_111313: n=50 p50=203.10 p95=476.13 OK
- labA_cpu_vs_gpu_pytorch_rocm_gpu_20261108_110918: n=50 p50=13.46 p95=25.64 OK
- labA_cpu_vs_gpu_pytorch_rocm_gpu_20261108_111140: n=50 p50=13.53 p95=15.12 OK
- labA_cpu_vs_gpu_pytorch_rocm_gpu_20261108_111315: n=50 p50=14.24 p95=15.73 OK
- labB_runtime_cpu_openvino_cpu_20261108_110944: n=50 p50=99.40 p95=106.08 OK
- labB_runtime_cpu_openvino_cpu_20261108_111149: n=50 p50=86.79 p95=93.85 OK
- labB_runtime_cpu_openvino_cpu_20261108_111340: n=50 p50=87.22 p95=91.67 OK
- labB_runtime_cpu_pytorch_cpu_20261108_110936: n=50 p50=199.39 p95=455.08 OK
- labB_runtime_cpu_pytorch_cpu_20261108_111205: n=50 p50=198.54 p95=440.56 OK
- labB_runtime_cpu_pytorch_cpu_20261108_111332: n=50 p50=187.67 p95=441.02 OK
- labC_int8_openvino_fp32_20261108_111041: n=50 p50=98.88 p95=102.13 OK
- labC_int8_openvino_fp32_20261108_111215: n=50 p50=98.30 p95=103.98 OK
- labC_int8_openvino_fp32_20261108_111349: n=50 p50=88.04 p95=92.97 OK
- labC_int8_openvino_int8_20261108_111049: n=50 p50=79.18 p95=83.73 OK
- labC_int8_openvino_int8_20261108_111222: n=50 p50=78.14 p95=83.43 OK
- labC_int8_openvino_int8_20261108_111355: n=50 p50=68.36 p95=73.56 OK
- labD_batch_batch1_e2e_20261108_111110: n=50 p50=13.43 p95=13.89 OK
- labD_batch_batch1_e2e_20261108_111243: n=50 p50=13.58 p95=14.86 OK
- labD_batch_batch1_e2e_20261108_111418: n=50 p50=13.75 p95=15.59 OK
- labD_batch_batch1_forward_20261108_111057: n=50 p50=10.51 p95=11.77 OK
- labD_batch_batch1_forward_20261108_111230: n=50 p50=10.50 p95=12.94 OK
- labD_batch_batch1_forward_20261108_111404: n=50 p50=10.74 p95=11.51 OK
- labD_batch_batch8_e2e_20261108_111112: n=50 p50=79.99 p95=83.29 OK
- labD_batch_batch8_e2e_20261108_111245: n=50 p50=80.04 p95=82.88 OK
- labD_batch_batch8_e2e_20261108_111419: n=50 p50=82.45 p95=100.76 OK
- labD_batch_batch8_forward_20261108_111103: n=50 p50=55.10 p95=55.43 OK
- labD_batch_batch8_forward_20261108_111236: n=50 p50=55.22 p95=61.04 OK
- labD_batch_batch8_forward_20261108_111410: n=50 p50=55.31 p95=56.02 OK
- labD_resolution_imgsz512_20261108_111054: n=50 p50=13.28 p95=20.04 OK
- labD_resolution_imgsz512_20261108_111227: n=50 p50=13.50 p95=14.22 OK
- labD_resolution_imgsz512_20261108_111401: n=50 p50=14.41 p95=21.07 OK
- labD_resolution_imgsz640_20261108_111052: n=50 p50=13.49 p95=14.68 OK
- labD_resolution_imgsz640_20261108_111225: n=50 p50=13.70 p95=14.96 OK
- labD_resolution_imgsz640_20261108_111358: n=50 p50=13.68 p95=16.36 OK
- labE_pipeline_async_pipeline_20261108_111121: n=100 p50=598.41 p95=1127.09 OK
- labE_pipeline_async_pipeline_20261108_111254: n=100 p50=603.53 p95=1133.92 OK
- labE_pipeline_async_pipeline_20261108_111429: n=100 p50=610.81 p95=1149.68 OK
- labE_pipeline_serial_20261108_111119: n=100 p50=13.50 p95=15.03 OK
- labE_pipeline_serial_20261108_111253: n=100 p50=13.86 p95=16.54 OK
- labE_pipeline_serial_20261108_111427: n=100 p50=13.96 p95=22.41 OK
- headline labA.speedup_x = 14.14x verified
- headline labB.speedup_x = 2.28x verified
- headline labC.speedup_x = 1.26x verified
- headline labD_batch.fwd_gain_x = 1.52x verified
- labE parity: 100/100 files, min IoU 1.0, max dConf 0.0
- headline labB.map_pytorch = 0.6398 matches maps
- headline labB.map_openvino = 0.6349 matches maps
- headline labC.map_fp32 = 0.6349 matches maps
- headline labC.map_int8 = 0.5682 matches maps
- headline labD_res.map_640 = 0.6398 matches maps
- headline labD_res.map_512 = 0.6159 matches maps

## Recomputed headline ratios

- labA.speedup_x = 14.135
- labB.speedup_x = 2.276
- labC.speedup_x = 1.258
- labD_batch.fwd_gain_x = 1.522

## Problems

NONE — all hashes, stats, invariants, ratios and parity checks verified independently.
