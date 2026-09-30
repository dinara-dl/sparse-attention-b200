"""Prepared FP4 proxies -> tile IDs -> dense13-line p03 GQA attention."""

from functools import lru_cache

from .fp4 import _validate_lengths, fp4_indexer


@lru_cache(maxsize=1)
def _load_attention():
    from ..progression import stage_functions
    return stage_functions("p03")[0]


@lru_cache(maxsize=1)
def _load_attention_out():
    from ..progression import stage_functions
    return stage_functions("p03")[1]


def _validate_inputs(q, k, v, inputs):
    import torch

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if (not tensor.is_cuda or tensor.device != q.device
                or tensor.dtype != torch.bfloat16 or tensor.ndim != 4
                or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous 4D BF16 on q's CUDA device")
        if tensor.requires_grad:
            raise ValueError("indexed_attention is forward-only; gradients are unsupported")
    batch, hq, seq_q, dim = q.shape
    if k.shape != v.shape or k.shape[0] != batch:
        raise ValueError("k/v shapes and q/k/v batch dimensions must match")
    hkv, seq_kv = k.shape[1:3]
    if dim != 128 or k.shape[3] != 128:
        raise ValueError("attention head dimension must be 128")
    if hkv <= 0 or hq % hkv or hq // hkv not in (2, 4, 8, 16):
        raise ValueError("supported GQA ratios Hq/Hkv are 2, 4, 8, and 16")
    _validate_lengths(batch, seq_q, seq_kv)
    if max(batch * hq * seq_q, batch * hkv * seq_kv) > 2**31 - 1:
        raise ValueError("flattened attention row counts must fit int32")
    expected = {"batch_size": batch, "hkv": hkv, "len_q": seq_q, "len_kv": seq_kv}
    for name, value in expected.items():
        if inputs[name] != value:
            raise ValueError(f"indexer {name} must match attention: expected {value}")
    if inputs["q"].device != q.device or inputs["k"].device != q.device:
        raise ValueError("prepared FP4 inputs and attention must share a CUDA device")


def indexed_attention(q, k, v, inputs):
    """Return (O, selected) from prepared proxies and full BF16 Q/K/V.

    Q is [B,Hq,Sq,128], K/V are [B,Hkv,Skv,128]. Inputs must come from
    prepare_fp4_inputs with the same B/Hkv/Sq/Skv and CUDA device.
    Selection is shared by every query head in a GQA group and by the 128
    queries in a tile. No causal mask, learned projections or backward.
    """
    import torch

    _validate_inputs(q, k, v, inputs)
    attention = _load_attention()
    with torch.cuda.device(q.device):
        selected = fp4_indexer(inputs)
        # Same-stream ordering makes the GPU-written IDs ready for p03.
        output = attention(q, k, v, selected)
    return output, selected
