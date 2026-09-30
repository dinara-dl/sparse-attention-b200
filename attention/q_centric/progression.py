"""Load dense14 main p02/p03 plus archived p01 for API compatibility/checks."""

from functools import lru_cache, partial
from pathlib import Path

try:
    from .checkpoints import PROGRESSION_SOURCES, PROGRESSION_VARIANTS
except ImportError:
    from checkpoints import PROGRESSION_SOURCES, PROGRESSION_VARIANTS

ROOT = Path(__file__).resolve().parents[1]
# cpp_extension derives object names from source basenames, not full paths.
BUILD_SOURCES = {**PROGRESSION_SOURCES, "p01": "compat/p01_dense13.cu"}
SOURCE_FILES = (
    "sparse/checkpoints.py", "sparse/progression.py", "sparse/progression_attention.cpp",
    *(f"sparse/{name}" for name in PROGRESSION_SOURCES.values()),
    "sparse/compat/p01_dense13.cu",
    "common.cuh", "approximations.cuh", "smem_layout_swizzle128.cuh",
)


@lru_cache(maxsize=1)
def load_ops():
    import torch
    from torch.utils.cpp_extension import load

    if not hasattr(torch.ops.xomtention_progression, "attention"):
        load(
            name="xomtention_dense14_progression_v1",
            sources=[str(ROOT / "sparse/progression_attention.cpp"),
                     *(str(ROOT / "sparse" / name) for name in BUILD_SOURCES.values())],
            extra_include_paths=[str(ROOT)], extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v",
                              "-gencode=arch=compute_100a,code=sm_100a"],
            extra_ldflags=["-lcuda"], is_python_module=False, verbose=True,
        )
    return torch.ops.xomtention_progression


def stage_functions(version="p03"):
    """Return allocating and preallocated callables for prepared selected IDs."""
    if version not in PROGRESSION_VARIANTS:
        raise ValueError(f"Expected one of {tuple(PROGRESSION_VARIANTS)}, got {version!r}")
    ops = load_ops()
    variant = PROGRESSION_VARIANTS[version]
    return partial(ops.attention, variant=variant), partial(ops.attention_out, variant=variant)
