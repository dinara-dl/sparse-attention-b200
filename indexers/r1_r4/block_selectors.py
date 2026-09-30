"""Explicit selector algorithms; no learned index projections or stdlib shadowing."""
import sys
from pathlib import Path
import torch

import gpu_ops as ops


class SummarySelector:
    def __init__(self, q, k, representatives):
        b, hq, sq, d = q.shape
        if hq != 16 or k.shape[:2] != (b, 1) or d != 128 or sq % 128:
            raise ValueError("this experiment deliberately uses H16/1, D128, full Q128 tiles")
        nt = k.shape[2] // 128
        if representatives not in (1, 4) or nt not in (16, 32, 64, 128) or k.shape[2] % 128:
            raise ValueError("R1/R4 and 16/32/64/128 complete KV blocks only")
        self.q, self.k, self.r, self.b, self.qt, self.nt = q, k, representatives, b, sq // 128, nt
        r = self.r
        self.qd = torch.empty((b, 16, self.qt, r, 128), dtype=torch.bfloat16, device=q.device)
        self.kd = torch.empty((b, 1, nt, r, 128), dtype=torch.bfloat16, device=q.device)
        self.raw = torch.empty((b, 16 * self.qt * r, nt * r), dtype=torch.float32, device=q.device)
        self.scores = torch.empty((b, 1, self.qt, nt), dtype=torch.float32, device=q.device)
        self.ids = torch.empty((b, 1, self.qt, 16), dtype=torch.int32, device=q.device)
        self.name = f"mean{r}_bf16"

    def prepare(self):
        ops.pool_blocks[(self.b * 16 * self.qt * self.r,)](
            self.q, self.qd, 16, self.qt, self.r, *self.q.stride()[:3], num_warps=8)
        ops.pool_blocks[(self.b * self.nt * self.r,)](
            self.k, self.kd, 1, self.nt, self.r, *self.k.stride()[:3], num_warps=8)

    def score(self):
        m, n = 16 * self.qt * self.r, self.nt * self.r
        ops.descriptor_dot[((m + 31) // 32, (n + 63) // 64, self.b)](
            self.qd, self.kd, self.raw, m, n, num_warps=4)

    def topk(self):
        ops.reduce_topk[(self.b * self.qt,)](self.raw, self.scores, self.ids,
                                          self.qt, self.nt, self.r, num_warps=8)

    def run(self):
        self.prepare()
        self.score()
        self.topk()
        return self.ids


class PreparedFP4Selector:
    """Unchanged existing FP4 API on prepared synthetic proxies: speed control only."""
    name = "fp4_prepared_synthetic"

    def __init__(self, here, batch, sq, skv, seed):
        sys.path.insert(0, str(Path(here) / "frozen"))
        from sparse.fp4_indexer.fp4 import make_synthetic_fp4_inputs, fp4_block_scores
        from sparse.fp4_indexer import tile_topk_out
        self.inputs = make_synthetic_fp4_inputs(batch_size=batch, hkv=1, len_q=sq,
                                                 len_kv=skv, seed=seed)
        self.scorer, self.tile_topk = fp4_block_scores, tile_topk_out
        self.ids = torch.empty((batch, 1, sq // 128, 16), dtype=torch.int32, device="cuda")
        self.raw = None

    def score(self):
        self.raw = self.scorer(self.inputs)

    def topk(self):
        self.tile_topk(self.raw, self.ids, batch_size=self.inputs["batch_size"], len_q=self.inputs["len_q"])

    def run(self):
        self.score()
        self.topk()
        return self.ids
