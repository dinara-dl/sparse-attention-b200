"""Prepared NVFP4 inputs -> public MSA scorer -> our query-tile TopK16."""

from functools import lru_cache
from pathlib import Path
import sys


MSA_COMMIT = "80434d7f67877c6570ca19cac444b84bc9855dac"
MSA_CUTE_DIR = (
    Path(__file__).resolve().parents[2]
    / "reference_repos/minimax_msa/python/fmha_sm100/cute"
)


@lru_cache(maxsize=1)
def _load_scorer():
    if not (MSA_CUTE_DIR / "fp4_indexer_interface.py").is_file():
        raise RuntimeError(f"MiniMax FP4 scorer is missing at {MSA_CUTE_DIR}")
    sys.path.insert(0, str(MSA_CUTE_DIR))
    from fp4_indexer_interface import (
        fp4_indexer_block_scores,
        fp4_indexer_reorder_scales_for_mma_cute,
    )

    return fp4_indexer_block_scores, fp4_indexer_reorder_scales_for_mma_cute


def _validate_lengths(batch_size, len_q, len_kv):
    if batch_size <= 0 or len_q <= 0 or len_q % 128:
        raise ValueError("B must be positive and Sq a positive multiple of 128")
    if len_kv % 128 or not 2048 <= len_kv <= 16384:
        raise ValueError("Skv must contain 16 to 128 complete KV blocks")
    if max(batch_size * len_q, batch_size * len_kv) > 2**31 - 1:
        raise ValueError("flattened sequence lengths must fit int32")


def prepare_fp4_inputs(q_fp4, k_fp4, q_scale, k_scale, *,
                       batch_size: int, len_q: int, len_kv: int):
    """Prepare already packed proxies; this does not quantize BF16 Q/K.

    Qidx has Hkv heads; Kidx has one shared head. Each logical D=128 vector
    occupies 64 uint8 bytes and has eight FP8 scales, one per 16 FP4 values.
    Scales must be finite and nonnegative. Reprepare if logical scales change.
    """
    import torch

    _validate_lengths(batch_size, len_q, len_kv)
    if q_fp4.ndim != 3 or q_fp4.shape[0] != batch_size * len_q:
        raise ValueError("Qidx must be [B*Sq,Hkv,64]")
    hkv = q_fp4.shape[1]
    if hkv <= 0 or q_fp4.shape[2] != 64:
        raise ValueError("Qidx must have positive Hkv and packed D=64 bytes")
    if batch_size * hkv * (len_q // 128) > 2**31 - 1:
        raise ValueError("query-tile count must fit int32")
    kv_tiles = len_kv // 128
    pages = batch_size * kv_tiles
    expected = (
        ("Qidx", q_fp4, (batch_size * len_q, hkv, 64), torch.uint8),
        ("Kidx", k_fp4, (pages, 1, 128, 64), torch.uint8),
        ("Q scales", q_scale, (batch_size * len_q, hkv, 8), torch.float8_e4m3fn),
        ("K scales", k_scale, (pages, 1, 128, 8), torch.float8_e4m3fn),
    )
    for name, tensor, shape, dtype in expected:
        if (not tensor.is_cuda or tensor.device != q_fp4.device
                or tensor.dtype != dtype or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on Qidx's CUDA device")

    _, reorder_scales = _load_scorer()
    with torch.cuda.device(q_fp4.device):
        q_mma, k_mma = reorder_scales(q_scale, k_scale, fp4_format="nvfp4")
        # Public reorder returns logical MMA views; the scorer wants storage order.
        q_mma = q_mma.permute(5, 2, 4, 0, 1, 3)
        k_mma = k_mma.permute(5, 2, 4, 0, 1, 3)
        requests = torch.arange(batch_size + 1, device=q_fp4.device, dtype=torch.int32)
        page_indices = torch.arange(pages, device=q_fp4.device, dtype=torch.int32)
    return {
        "q": q_fp4, "k": k_fp4,
        "q_scale": q_scale, "k_scale": k_scale,
        "q_scale_mma": q_mma, "k_scale_mma": k_mma,
        "cu_q": requests * len_q, "cu_k": requests * len_kv,
        "cu_pages": requests * kv_tiles, "page_indices": page_indices,
        "batch_size": batch_size, "hkv": hkv,
        "len_q": len_q, "len_kv": len_kv,
    }


def make_synthetic_fp4_inputs(*, batch_size: int, hkv: int, len_q: int,
                              len_kv: int, seed: int = 0,
                              scale_mode: str = "varied", device="cuda"):
    """Generate synthetic FP4 proxies, independent of the main attention Q/K/V."""
    import torch

    _validate_lengths(batch_size, len_q, len_kv)
    if hkv <= 0 or scale_mode not in ("ones", "varied"):
        raise ValueError("Hkv must be positive; scale_mode must be ones or varied")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("FP4 input preparation requires CUDA")
    generator = torch.Generator(device=device).manual_seed(seed)
    pages = batch_size * (len_kv // 128)
    q = torch.randint(256, (batch_size * len_q, hkv, 64), device=device,
                      dtype=torch.uint8, generator=generator)
    k = torch.randint(256, (pages, 1, 128, 64), device=device,
                      dtype=torch.uint8, generator=generator)

    def scales(shape):
        if scale_mode == "ones":
            values = torch.ones(shape, device=device)
        else:
            choices = torch.tensor([0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0], device=device)
            indices = torch.randint(len(choices), shape, device=device, generator=generator)
            values = choices[indices]
        return values.to(torch.float8_e4m3fn)

    return prepare_fp4_inputs(
        q, k, scales((batch_size * len_q, hkv, 8)), scales((pages, 1, 128, 8)),
        batch_size=batch_size, len_q=len_q, len_kv=len_kv,
    )


def fp4_block_scores(inputs):
    """Return FP32 [Hkv,KV_tiles,B*Sq] scores from the unchanged MSA scorer."""
    import torch

    scorer, _ = _load_scorer()
    with torch.cuda.device(inputs["q"].device):
        return scorer(
            inputs["q"], inputs["k"], inputs["q_scale_mma"], inputs["k_scale_mma"],
            inputs["cu_q"], inputs["cu_k"], inputs["cu_pages"],
            max_seqlen_q=inputs["len_q"], max_seqlen_k=inputs["len_kv"],
            kv_indices=inputs["page_indices"], fp4_format="nvfp4",
            causal=False, scale_layout="preordered_mma",
        )


def fp4_indexer(inputs, selected=None):
    """Run FP4 scoring followed by our max-over-Q and TopK16; return only IDs."""
    from . import tile_topk, tile_topk_out

    scores = fp4_block_scores(inputs)
    kwargs = {"batch_size": inputs["batch_size"], "len_q": inputs["len_q"]}
    if selected is None:
        return tile_topk(scores, **kwargs)
    tile_topk_out(scores, selected, **kwargs)
    return selected
