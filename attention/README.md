# Attention kernels

Contract for both kernels: BF16, head dimension 128, non-causal forward, KV blocks of 128 tokens.

## q_centric/ (Chapters 2–3)

One CTA owns a Q tile and walks through its list of selected KV blocks.

| File | Role |
|---|---|
| `01_block_sparse.cu` | **p02, block-sparse.** The dense kernel with the KV loop changed to selected blocks. One CTA task = two neighboring Q128 tiles of one head, each with its own block list; both use one three-slot K/V ring. |
| `02_gqa_block_sparse.cu` | **p03, + GQA.** One Q128 tile of two heads from the same GQA group. The heads share the block list, so each selected K/V block is loaded once for the pair. This is the kernel compared with FA4 in Chapters 3–4. |
| `progression_attention.cpp` | PyTorch operator `xomtention_progression::attention`. Variant 1 = p02, variant 2 = p03. Variant 0 is an older archived kernel and is not included. |
| `progression.py`, `checkpoints.py` | Build and load the extension: `stage_functions("p02")`, `stage_functions("p03")`. |
| `progression_validation.py` | Independent FP32 reference and correctness checks (tolerance `0.02 + 0.02·|reference|`). |
| `progression_modal.py` | Modal job for the correctness gate and the CUDA Graph benchmark on B200. |
| `common.cuh`, `approximations.cuh`, `smem_layout_swizzle128.cuh` | Shared SM100 helpers from the dense kernel series. |
| `reference/00_dense14.cu` | The dense starting point. |
| `diffs/` | Readable diffs: dense → sparse, sparse → GQA. |
| `tests/test_schedule.py` | CPU model of the K/V ring schedule (not a CUDA test). |

Inputs: Q `[B, Hq, Sq, 128]`, K/V `[B, Hkv, Skv, 128]`, block IDs `int32 [B, Hkv, Sq/128, TopK]`.

## kv_centric/ (Chapter 5)

One CTA owns a KV block and processes the queries that selected it.
Every query has its own list of 16 blocks: `selected` is `int32 [B, Hkv, Sq, 16]`.

The folder names keep the original round numbers, because the code imports files by these paths.
`optimization_round11/` is the final version; the other rounds hold files it builds on.

Final variants (`optimization_round11/api11.py`):

| Variant | Partial outputs | Note |
|---|---|---|
| `tma8` | FP32 | Q and partials in token-major layout, gathered and stored with TMA. Left column of the time-split table in the post. |
| `tma8_bf16` | BF16 | Same kernel with BF16 partials. Right column of the time-split table. |
| `tma16_bf16` | BF16 | 16 chunks per task; the fastest variant (1.50 ms). |
| `split8`, `split8_bf16` | FP32, BF16 | Round 9 parents: two independent pipelines per CTA, older memory layout. |

The TMA path needs at least 8 query heads per KV head; with fewer, the kernel uses the round 9 copy path.

| File | Role |
|---|---|
| `optimization_round11/tma11_kernel.cu` | Final attention kernel: gather Q, QK, softmax, PV, partial O + LSE. |
| `optimization_round11/combine11.cu` | Combine for token-major partials. |
| `optimization_round11/api11.py`, `wrapper11.cpp` | Python API (`Prepared(q, k, v, selected, variant)`) and PyTorch binding. |
| `optimization_round9/split9_kernel.cu` | Two-pipeline kernel, parent of round 11. |
| `optimization_round8/combine8.cu`, `optimization_round1/combine_warp.cu` | Combine kernels for the round 9 variants (BF16 and FP32 partials). |
| `optimization_round4/gpu_metadata.cu`, `gpu_plan.py` | Reverse index (block → queries) and task split, built on the GPU. `planning.py`, `bench_api4.py`, `timing.py` are its helpers. |
| `optimization_round7/timing7.py` | CUDA Graph timing harness. |
| `primitives.cuh` | SM100 MMA, TMEM and TMA helpers. |
| `metadata.py` | Synthetic block lists (`independent`, `hot`, `identical`, …) and a CPU version of the reverse index. |
| `reference.py` | FP32 attention oracle and Combine reference. |
| `api.py`, `validate_gpu.py` | First-version API and the GPU correctness suite reused by later rounds (the first-version kernel itself is not included). |
| `REFERENCE_DECISION_RU.md` | Why the MSA dataflow was chosen and where the primitives come from (in Russian). |
