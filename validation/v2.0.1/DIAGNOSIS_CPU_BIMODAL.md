# R2.3 Diagnosis — OpenVINO CPU bimodal latency

Date: 2026-11-08T11:33:42 · 2x AMD EPYC 9334 (64C/128T) · cgroup quota 16.0 cores · shared host

## Finding

The v2.0.0 bimodal CPU latency (~26/75 ms strict alternation) is **container cgroup CPU throttling**, not noise:

- OpenVINO's LATENCY config defaults to one thread per visible core (64 here) while the container quota is 16 cores;
- an inference burst exhausts the 100 ms period budget, the kernel throttles the cgroup, and the NEXT inference pays the throttle window — producing deterministic fast/slow alternation;
- proof: the number of 'slow' samples equals the delta of `nr_throttled` in `/sys/fs/cgroup/cpu.stat` within each experiment.

## Experiments

| # | Setup | P50 ms | slow samples | Δnr_throttled | verdict |
|---|---|---|---|---|---|
| E | raw OV · default threads (64) | 55.2 | 16 | 16 | BIMODAL (throttled) |
| E | raw OV · 16 threads (=quota) | 34.8 | 0 | 6 | stable |
| E | ultralytics e2e · unpinned | 47.2 | 15 | 15 | BIMODAL (throttled) |
| E | ultralytics e2e · pinned 16 CPUs | 83.9 | 30 | 1 | BIMODAL (throttled) |

## Consequence for v2.0.1 benchmarks

- All CPU benchmark variants pin the process to 16 CPUs (re-picked quiet set per round, recorded per run) matching the quota;
- PyTorch CPU threads pinned to 8 (measured optimum under quota);
- host is SHARED with other tenants — load average and throttle counters are recorded around every campaign round and disclosed.

Raw samples: DIAGNOSIS_CPU_BIMODAL.json (same directory).
