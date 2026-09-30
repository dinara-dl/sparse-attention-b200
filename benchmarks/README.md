# Benchmarks

All runs: NVIDIA B200 through Modal, BF16, head dimension 128, KV blocks of 128 tokens, 16 blocks per list.
Timings are medians of CUDA Graph replays over several seeds and rounds; each folder's RESULTS file gives the exact protocol.

| Folder | Blog | What it measures | Start with |
|---|---|---|---|
| `ch3_q_centric_vs_fa4/` | Chapter 3 | p03 against FA4 sparse (best of `q_stage` 1/2 × PackGQA on/off) on identical Q/K/V and block IDs. Batch 2, 16 query heads per KV head, 4096 queries, 2K–16K keys. | `RESULTS.md`, `summary.json` |
| `ch4_selectors/` | Chapter 4 | R1, R4 and FP4 on their own, then the full selector → attention pipeline with p03 and with FA4. Includes the benchmark's p03/FA4 backends (`backends.py`, `harness.py`, `reference.py`). | `RESULTS.md`, `summary.json` |
| `ch5_kv_centric/round11_final/` | Chapter 5 | Final KV-centric variants. The confirmation run (3 seeds) gives the time-split table in the post (`tma8` and `tma8_bf16`). | `RESULTS_RU.md`, `results/run_003_confirmation.json` |
| `ch5_kv_centric/round7_profile/` | Chapter 5 | Ablation and Nsight Compute profile of an earlier kernel version: the "77% without any math" result. | `RESULTS_RU.md`, `results/run_001_profile.json` |
| `common_fa4_harness/` | Chapters 3–4 | Shared setup: FA4 adapter, block-list metadata, fairness controls and the Modal image with pinned FA4. | — |

Raw JSON files in each `results/` folder are the direct outputs of the Modal jobs;
`*_call.json` files record the Modal call IDs.
