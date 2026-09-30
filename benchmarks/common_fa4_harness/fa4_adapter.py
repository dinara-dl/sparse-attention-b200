"""Unmodified FA4 kernels; exact mask/layout adapters and a labeled Q-stage override.

Q-stage 1 is exposed through a scoped Python FwdConfig override, not a patched
CUDA kernel or modified reference checkout. Stock q_stage=2 remains a row.
"""
from contextlib import contextmanager
from dataclasses import replace

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute import interface as fa
from flash_attn.cute import utils

from metadata import tensor_metadata


@cute.jit
def folded_q128_mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    lookup = aux_tensors[0]
    qblock = cutlass.min(q_idx[0] // 128, lookup.shape[2] - 1)
    kblock = cutlass.min(kv_idx[0] // 128, lookup.shape[3] - 1)
    value = lookup[batch[0], 0, qblock, kblock]
    return utils.scalar_to_ssa(value, cutlass.Int32) != 0


@cute.jit
def native_g16_q128_mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    lookup = aux_tensors[0]
    qblock = cutlass.min(q_idx[0] // 128, lookup.shape[2] - 1)
    kblock = cutlass.min(kv_idx[0] // 128, lookup.shape[3] - 1)
    value = lookup[batch[0], head[0] // 16, qblock, kblock]
    return utils.scalar_to_ssa(value, cutlass.Int32) != 0


@contextmanager
def forward_configuration(q_stage, observations=None):
    original_config = fa._get_fwd_config
    original_constructor = fa.FlashAttentionForwardSm100

    def config(*args, **kwargs):
        result = original_config(*args, **kwargs)
        if q_stage is not None:
            result = replace(result, q_stage=q_stage)
        return result

    def constructor(*args, **kwargs):
        result = original_constructor(*args, **kwargs)
        if observations is not None:
            observations.append({name: getattr(result, name) for name in (
                "q_stage", "m_block_size", "n_block_size", "pack_gqa", "is_split_kv",
                "use_2cta_instrs", "use_clc_scheduler", "is_static_persistent",
            )})
        return result

    if q_stage is not None:
        fa._get_fwd_config = config
    if observations is not None:
        fa.FlashAttentionForwardSm100 = constructor
    try:
        yield
    finally:
        fa._get_fwd_config = original_config
        fa.FlashAttentionForwardSm100 = original_constructor


class Fa4Backend:
    def __init__(self, q, k, v, *, plan=None, q_stage=2, pack=True, folded=True):
        self.batch, self.hq, self.sq, self.dim = q.shape
        self.hkv, self.skv = k.shape[1:3]
        self.group = self.hq // self.hkv
        self.folded, self.q_stage, self.pack = folded, q_stage, pack
        self.is_dense = plan is None
        if folded:
            self.q = q.reshape(self.batch, self.hkv, self.group, self.sq, self.dim).permute(0, 1, 3, 2, 4).reshape(self.batch * self.hkv, self.sq, self.group, self.dim).contiguous()
            self.k = k.reshape(self.batch * self.hkv, self.skv, 1, self.dim).contiguous()
            self.v = v.reshape(self.batch * self.hkv, self.skv, 1, self.dim).contiguous()
        else:
            self.q = q.transpose(1, 2).contiguous()
            self.k = k.transpose(1, 2).contiguous()
            self.v = v.transpose(1, 2).contiguous()
        self.out = torch.empty_like(self.q)
        self.metadata, self.aux, self.mask = None, None, None
        self.metadata_stats = None
        if plan is not None:
            self.metadata, lookup = tensor_metadata(plan, hq=self.hq, folded=folded)
            self.metadata_stats = plan.stats()
            if self.metadata_stats["partial_block_count"]:
                self.mask = folded_q128_mask if folded else native_g16_q128_mask
                self.aux = [lookup]
        self.observations = []

    def run(self, *, observe=False):
        # q_stage=2 uses the public dispatch unchanged. Dense is always stock.
        override = 1 if not self.is_dense and self.q_stage == 1 else None
        with forward_configuration(override, self.observations if observe else None):
            return fa._flash_attn_fwd(
                self.q, self.k, self.v, out=self.out, softmax_scale=self.dim ** -0.5,
                causal=False, num_splits=1, pack_gqa=self.pack,
                block_sparse_tensors=self.metadata, mask_mod=self.mask,
                aux_tensors=self.aux, return_lse=False,
            )[0]

    def logical_output(self):
        if self.folded:
            return self.out.reshape(self.batch, self.hkv, self.sq, self.group, self.dim).permute(0, 1, 3, 2, 4).reshape(self.batch, self.hq, self.sq, self.dim)
        return self.out.transpose(1, 2)

    def description(self):
        expected_pack = self.pack and (self.is_dense or self.folded or self.hkv == 1)
        return {
            "fold_gqa_groups_into_batch": self.folded,
            "physical_q_shape": list(self.q.shape), "physical_k_shape": list(self.k.shape),
            "logical_q_shape": [self.batch, self.hq, self.sq, self.dim],
            "q_strides": list(self.q.stride()), "k_strides": list(self.k.stride()),
            "q_stage": self.q_stage, "pack_gqa_requested": self.pack,
            "pack_gqa_expected": expected_pack, "num_splits": 1,
            "q_stage1_python_config_override": not self.is_dense and self.q_stage == 1,
            "observed_compiled_configurations": self.observations,
            "metadata": self.metadata_stats,
        }
