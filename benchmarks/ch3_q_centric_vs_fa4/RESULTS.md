# Current p03 vs pinned FA4: measured results

B200; B2, Hq/Hkv=16/1, Sq4096, D128, BF16, non-causal, TopK16, KV128.
Current `sparse/02_gqa_block_sparse.cu`, through the production p03 loader.
CUDA SHA-256: `0b224b2ce92d2192a5dd745cb607e04a73fb4010ebdc7373e3ab350d28268127`.
FA4: `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`; source tree unchanged.

Latency in microseconds: median of per-seed medians (three rounds each; seeds17/29/41).
Best FA4 is the lowest aggregate latency among four tested sparse / four dense configurations.
Ratios are medians of within-seed ratios for that selected configuration, not ratios across GPU allocations.

| Skv | Mask | Selected % | Current p03 µs | Best FA4 sparse µs | FA4 sparse / p03 | Best FA4 dense µs | FA4 dense / p03 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 2048 | all | 100.0 | 136.63 | 120.12 | 0.879× | 118.15 | 0.864× |
| 4096 | independent | 50.0 | 137.81 | 157.00 | 1.139× | 228.20 | 1.656× |
| 4096 | paired | 50.0 | 136.90 | 122.08 | 0.892× | 227.79 | 1.664× |
| 8192 | independent | 25.0 | 138.06 | 157.23 | 1.138× | 448.67 | 3.247× |
| 8192 | paired | 25.0 | 136.99 | 121.83 | 0.889× | 449.02 | 3.278× |
| 16384 | independent | 12.5 | 140.27 | 159.95 | 1.139× | 894.60 | 6.379× |
| 16384 | paired | 12.5 | 138.76 | 123.65 | 0.891× | 894.35 | 6.445× |

## All tested configurations

`q1` is an explicit Python launch override; Q2 partial blocks use the vector32 exact mask.
Dense `1cta` explicitly disables 2CTA; `autocta` retains the pinned FA4 choice.

| Skv | Mask | p03 | Q1 unpack | Q1 pack | Q2 unpack | Q2 pack | Dense unpack 1CTA | Dense pack 1CTA | Dense unpack auto | Dense pack auto |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | all | 136.63 | 152.45 | 157.51 | 120.12 | 123.29 | 118.15 | 121.34 | 144.36 | 144.32 |
| 4096 | independent | 137.81 | 157.00 | 158.26 | 250.63 | 233.14 | 228.20 | 234.67 | 280.45 | 280.32 |
| 4096 | paired | 136.90 | 153.65 | 157.75 | 122.08 | 122.94 | 227.79 | 234.05 | 280.25 | 280.78 |
| 8192 | independent | 138.06 | 157.23 | 157.76 | 316.79 | 289.98 | 448.67 | 463.79 | 549.28 | 550.43 |
| 8192 | paired | 136.99 | 153.25 | 157.55 | 121.83 | 123.26 | 449.02 | 463.32 | 550.27 | 550.61 |
| 16384 | independent | 140.27 | 159.95 | 161.13 | 347.41 | 308.39 | 894.60 | 921.94 | 1097.74 | 1097.44 |
| 16384 | paired | 138.76 | 156.42 | 160.45 | 123.65 | 124.57 | 894.35 | 920.94 | 1096.39 | 1097.60 |

## Validation and timing

- 5 full correctness-gate cases; 21 measured input/mask cases; 567 backend round timings.
- Full FP32 references for all outputs before timing and after replay. Maximum absolute reference error across these runs: 0.02260780.
- p03 alone: maximum absolute reference error 0.01999950 in the gate (including scale4), 0.00162134 on the timed scale1 inputs.
- Tolerance is the combined criterion abs(error) <= 0.02 + 0.02*abs(reference), not a standalone 0.02 maximum.
- Largest three-round min/max spread relative to the median: p03 1.195%; any backend 1.195%.
- Graph8, warmup250ms, measurement1000ms, three rounds and seeds17/29/41; outputs preallocated.
- Same GPU for all backends/seeds/patterns at one length; different lengths may use different physical B200s.
- Paired and independent patterns use bit-identical Q/K/V for each seed. Full/partial metadata preserves the exact mask.
- Same-buffer steady state, not cold-cache serving. Inputs, indexer, metadata, layouts, allocations, compilation and reference checks excluded.
- Dense has a different mask except the 2K all-block control. No model-quality, end-to-end selector, sanitizer or profiler claim.
- All first-party GPU functions retain timeout=8*60; CUDA sources and historical results are untouched.

## Evidence

- [20260918T123745785531Z_gate_4096.json](results/20260918T123745785531Z_gate_4096.json): 72.4s remote; app `ap-cjJV4XBjpnlScRgJSwedl8`.
- [20260918T123955572023Z_performance_2048.json](results/20260918T123955572023Z_performance_2048.json): 108.0s remote; app `ap-xp3KwzxgMsqdWldQOZV8zB`.
- [20260918T124200951789Z_performance_4096.json](results/20260918T124200951789Z_performance_4096.json): 205.9s remote; app `ap-J57J8zoRlLvYHspxiWNk1r`.
- [20260918T124556513319Z_performance_8192.json](results/20260918T124556513319Z_performance_8192.json): 213.2s remote; app `ap-qeB7ysKLpjUWE0oDLYrWYt`.
- [20260918T124956164843Z_performance_16384.json](results/20260918T124956164843Z_performance_16384.json): 214.4s remote; app `ap-fyf2NM4tgjjtUcP3R6FTfP`.
