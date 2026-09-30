# Fresh selector benchmark: current p03 vs FA4 sparse

All times in microseconds; lower is better. Median across three input seeds.

| KV | Selector | Selector only | p03 attention only | Selector → p03 | Selector → best FA4 | p03 speedup |
|---|---|---:|---:|---:|---:|---:|
| 4K | R1 | 20.73 | 140.72 | 155.72 | 174.66 | 1.122× |
| 4K | R4 | 47.84 | 140.56 | 175.00 | 195.79 | 1.119× |
| 4K | FP4* | 17.66 | 140.54 | 154.67 | 173.60 | 1.122× |
| 8K | R1 | 22.01 | 137.70 | 154.33 | 173.37 | 1.123× |
| 8K | R4 | 53.55 | 137.73 | 178.44 | 203.45 | 1.140× |
| 8K | FP4* | 28.92 | 137.92 | 162.50 | 186.29 | 1.146× |
| 16K | R1 | 22.09 | 137.56 | 154.50 | 174.43 | 1.129× |
| 16K | R4 | 60.73 | 137.65 | 187.04 | 211.43 | 1.130× |
| 16K | FP4* | 48.38 | 137.49 | 180.43 | 208.77 | 1.157× |

*FP4 starts from independent prepared synthetic proxies: projection, quantization and preparation excluded.
Its boundary and information differ from R1/R4; this is not a selector-quality ranking.

Full pipelines are measured directly, not sums of isolated timings. FA4 Q2 metadata is included.
Common BHSD inputs with zero-copy BSHD views for FA4. See README for the full protocol.

## Best tested FA4 configurations

- 4K R1: fa4_q1_pack_pipeline; largest within-case round spread 1.18%.
- 4K R4: fa4_q1_unpack_pipeline; largest within-case round spread 0.23%.
- 4K FP4*: fa4_q1_pack_pipeline; largest within-case round spread 0.92%.
- 8K R1: fa4_q1_unpack_pipeline; largest within-case round spread 0.88%.
- 8K R4: fa4_q1_unpack_pipeline; largest within-case round spread 0.29%.
- 8K FP4*: fa4_q1_unpack_pipeline; largest within-case round spread 0.48%.
- 16K R1: fa4_q1_unpack_pipeline; largest within-case round spread 0.84%.
- 16K R4: fa4_q1_unpack_pipeline; largest within-case round spread 0.24%.
- 16K FP4*: fa4_q1_unpack_pipeline; largest within-case round spread 0.50%.

## Raw result files

- `results/20260919T190801846116Z_performance_4096_seed0.json`
- `results/20260919T191243788178Z_performance_8192_seed0.json`
- `results/20260919T191732527995Z_performance_16384_seed0.json`
