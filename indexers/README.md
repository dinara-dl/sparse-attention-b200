# Block selectors (indexers), Chapter 4

Every selector returns block IDs `int32 [B, 1, Sq/128, 16]`: 16 KV blocks for each 128-query tile and GQA group.
The attention kernel then reads the original BF16 Q/K/V; the vectors below are used only for selection.

## r1_r4/

| File | Role |
|---|---|
| `block_selectors.py` | `SummarySelector(q, k, representatives)`: R1 with `representatives=1`, R4 with `representatives=4`. Averages the 128 queries of a tile and the 128 keys of a block (R1), or four groups of 32 (R4), takes dot products per query head, keeps the maximum over representatives and heads, and selects TopK16. `PreparedFP4Selector` wraps the FP4 path below. |
| `gpu_ops.py` | Triton kernels: `pool_blocks` (averaging), `descriptor_dot` (scores), `reduce_topk` (maximum + TopK16). `q2_metadata` converts block IDs into FA4's block-sparse metadata for the benchmark. |

## fp4/

| File | Role |
|---|---|
| `fp4.py` | Builds synthetic NVFP4 index vectors (`Qidx`, `Kidx`) with scales and calls the MiniMax scorer: for each query and KV block, the maximum score over the block's 128 keys. |
| `tile_topk.cu`, `wrapper.cpp`, `__init__.py` | Our tile adapter: maximum over the 128 queries of a tile, then TopK16 blocks. |
| `pipeline.py` | Scorer → TopK → attention in one call. |
| `fp4_reference.py`, `reference.py` | PyTorch references for the checks. |
| `README.md` | Original notes; its links point to the original workspace. |
| `vendor_minimax_msa/` | The six scorer files from MiniMax MSA (commit `80434d7f`), MIT License. |

All inputs are synthetic. There are no learned index projections, the FP4 timing excludes input
preparation and quantization, and selection quality on a real model was not evaluated.
