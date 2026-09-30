"""Query-tile max and TopK16 adapter. The FP4 pipeline lives in fp4.py."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_extension():
    import torch
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent
    load(
        name="xomtention_tile_topk",
        sources=[str(source_dir / "wrapper.cpp"), str(source_dir / "tile_topk.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "-lineinfo", "--ptxas-options=-v",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        is_python_module=False,
        verbose=True,
    )
    return torch.ops.xomtention_tile_topk


def tile_topk(scores, *, batch_size: int, len_q: int):
    """Return int32 [B,Hkv,Sq/128,16] IDs from finite FP32 CUDA scores.

    Scores must be contiguous [Hkv,KV_tiles,B*Sq]. Equal scores prefer the
    smaller block ID. The first call builds and loads the CUDA extension.
    """
    return _load_extension().tile_topk(scores, batch_size, len_q)


def tile_topk_out(scores, selected, *, batch_size: int, len_q: int):
    """Write IDs into a preallocated tensor on the current CUDA stream."""
    _load_extension().tile_topk_out(scores, selected, batch_size, len_q)
