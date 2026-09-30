"""GPU-updated masks, common B/H/S/D inputs, no hidden stale contiguous copies."""
import torch
from flash_attn.cute import interface as fa
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

from fa4_adapter import forward_configuration
from controls import block_uniform_vector32, disable_two_cta
import gpu_ops as ops


class DynamicFa4:
    def __init__(self, q, k, v, ids=None, stage=1, pack=False, dense=False, one_cta=True):
        self.b, self.hq, self.sq, _ = q.shape
        self.qt, self.nt = self.sq // 128, k.shape[2] // 128
        if self.hq != 16 or k.shape[1] != 1 or self.qt % 2:
            raise ValueError("H16/1 with complete Q256 pairs only")
        self.q, self.k, self.v = (x.transpose(1, 2) for x in (q, k, v))
        # FA4 accepts outer strides; these are views of the SAME changing inputs.
        self.out = torch.empty(self.q.shape, dtype=q.dtype, device=q.device)
        self.ids, self.stage, self.pack, self.dense, self.one_cta = ids, stage, pack, dense, one_cta
        self.metadata, self.aux, self.mask = None, None, None
        self.observations = []
        if not dense:
            if stage == 1:
                self.fc = torch.full((self.b, 1, self.qt), 16, device=q.device, dtype=torch.int32)
                self.pc = torch.zeros_like(self.fc)
                self.fi, self.pi = ids, torch.zeros_like(ids)
            elif stage == 2:
                self.fc = torch.empty((self.b, 1, self.qt // 2), device=q.device, dtype=torch.int32)
                self.pc = torch.empty_like(self.fc)
                self.fi = torch.empty((self.b, 1, self.qt // 2, 32), device=q.device, dtype=torch.int32)
                self.pi = torch.empty_like(self.fi)
                self.lookup = torch.empty((self.b, 1, self.qt, self.nt), device=q.device, dtype=torch.uint8)
                self.aux, self.mask = [self.lookup], block_uniform_vector32
            else:
                raise ValueError(stage)
            self.metadata = BlockSparseTensorsTorch(self.pc, self.pi, self.fc, self.fi,
                                                    block_size=(128 * stage, 128))

    def prepare(self):
        if not self.dense and self.stage == 2:
            ops.q2_metadata[(self.b * self.qt // 2,)](
                self.ids, self.fc, self.fi, self.pc, self.pi, self.lookup, self.qt, self.nt, num_warps=4)

    def run(self, observe=False):
        override = 1 if not self.dense and self.stage == 1 else None
        with disable_two_cta(self.dense and self.one_cta), forward_configuration(override, self.observations if observe else None):
            fa._flash_attn_fwd(self.q, self.k, self.v, out=self.out, causal=False,
                             softmax_scale=128 ** -0.5, pack_gqa=self.pack, num_splits=1,
                             block_sparse_tensors=self.metadata, mask_mod=self.mask,
                             aux_tensors=self.aux, return_lse=False)
        return self.out

    def logical_output(self):
        return self.out.transpose(1, 2)

    def description(self):
        return {"q_stage": self.stage, "pack_gqa": self.pack, "num_splits": 1,
                "dense": self.dense, "dense_force_1cta": self.one_cta,
                "q_shape": list(self.q.shape), "q_strides": list(self.q.stride()),
                "k_strides": list(self.k.stride()), "common_input_zero_copy_views": True,
                "q1_metadata": "IDs alias full-block indices; constant counts; zero GPU adapter launches",
                "q2_metadata": "one GPU intersection/union kernel, no CPU IDs read during pipeline",
                "q2_mask_callback_always_present": not self.dense and self.stage == 2,
                "observed_compiled_configurations": self.observations}
