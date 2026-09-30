"""Public correctness-first API. All local build artifacts stay in this directory."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from metadata import BLOCK, GROUPS, TOPK, build_plan, validate_plan, validate_selected

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/B200 required; CPU reference tests are independent")
    from torch.utils.cpp_extension import load
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    load(name="xomtention_kv0", sources=[str(ROOT / "wrapper.cpp"), str(ROOT / "kernel.cu")],
         build_directory=str(build_dir), extra_include_paths=[str(ROOT)],
         extra_cflags=["-O2"],
         extra_cuda_cflags=["-O2", "-lineinfo", "--ptxas-options=-v",
                           "-gencode=arch=compute_100a,code=sm_100a"],
         extra_ldflags=["-lcuda"], is_python_module=False, verbose=True)
    return torch.ops.xomtention_kv0


def check_inputs(q, k, v, selected):
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must be rank-4 tensors")
    if any(x.dtype != torch.bfloat16 or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("Q/K/V must be contiguous BF16")
    if any(x.device != q.device for x in (k, v)):
        raise ValueError("Q/K/V must share a device")
    if any(x.requires_grad for x in (q, k, v)):
        raise ValueError("forward only; gradients are outside v0")
    b, hq, sq, d = q.shape
    if min(q.shape) <= 0 or min(k.shape) <= 0 or d != 128 or k.shape[-1] != 128:
        raise ValueError("positive shapes with D128 required")
    if k.shape != v.shape or k.shape[0] != b:
        raise ValueError("K/V and batch shapes must match")
    hkv, skv = k.shape[1:3]
    if hq % hkv or hq // hkv not in GROUPS:
        raise ValueError("G must be one of 1,2,4,8,16")
    if skv % BLOCK or skv < TOPK * BLOCK:
        raise ValueError("Skv must be >=2048 and divisible by 128")
    validate_selected(selected, skv // BLOCK)
    if tuple(selected.shape) != (b, hkv, sq, TOPK):
        raise ValueError("selected must be [B,Hkv,Sq,16], not [B,Hkv,Q_tiles,16]")


@torch.no_grad()
def attention(q, k, v, selected, *, plan=None, return_debug=False):
    check_inputs(q, k, v, selected)
    if q.device.type != "cuda":
        raise ValueError("attention is the CUDA backend; use attention_reference on CPU")
    group = q.shape[1] // k.shape[1]
    if plan is None:
        plan = build_plan(selected, k.shape[2] // BLOCK, group)
    if plan.group_size != group or plan.kv_blocks != k.shape[2] // BLOCK:
        raise ValueError("plan does not match Q/K geometry")
    validate_plan(plan, selected)  # Deliberately complete audit in v0.
    tensors = [t.to(q.device) for t in (plan.tasks, plan.query_ids, plan.ranks)]
    out, lse, partial_o, partial_lse = load_extension().forward(q, k, v, *tensors)
    if return_debug:
        return out, lse, partial_o, partial_lse
    return out
