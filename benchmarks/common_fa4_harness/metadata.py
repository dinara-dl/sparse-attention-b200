"""CPU preparation of compact, exact FA4 metadata; never in core timing.

No token-by-token Sq*Skv mask is constructed. The optional lookup table has
one byte per Q128/KV128 block pair. Full and partial index lists are disjoint.
"""

from dataclasses import dataclass
import math
import random

BLOCK = 128
TOPK = 16


def make_lists(batch, hkv, q_tiles, kv_tiles, pattern, seed):
    if min(batch, hkv, q_tiles) < 1 or kv_tiles < TOPK:
        raise ValueError("positive dimensions and at least sixteen KV blocks required")
    if pattern not in ("all", "paired", "independent", "disjoint", "hot"):
        raise ValueError(pattern)
    if pattern == "all" and kv_tiles != TOPK:
        raise ValueError("all-block control keeps TopK16 and requires Skv=2048")
    if pattern == "disjoint" and kv_tiles < 2 * TOPK:
        raise ValueError("disjoint TopK16 pairs require at least 32 KV blocks")
    rng = random.Random(seed)
    result = []
    for _ in range(batch):
        heads = []
        for _ in range(hkv):
            rows = []
            for qi in range(q_tiles):
                if pattern == "all":
                    row = list(range(TOPK))
                elif pattern == "paired" and qi % 2:
                    row = list(rows[-1])
                elif pattern == "disjoint" and qi % 2:
                    row = rng.sample([x for x in range(kv_tiles) if x not in rows[-1]], TOPK)
                elif pattern == "hot":
                    row = list(range(8)) + rng.sample(list(range(8, kv_tiles)), 8)
                else:
                    row = rng.sample(list(range(kv_tiles)), TOPK)
                rows.append(sorted(row))
            heads.append(rows)
        result.append(heads)
    return result


@dataclass
class Plan:
    selected: list
    kv_tiles: int
    q_stage: int
    full: list
    partial: list
    allowed: list

    @property
    def batch(self):
        return len(self.selected)

    @property
    def hkv(self):
        return len(self.selected[0])

    @property
    def q_tiles(self):
        return len(self.selected[0][0])

    def stats(self):
        nfull = sum(len(row) for b in self.full for h in b for row in h)
        npartial = sum(len(row) for b in self.partial for h in b for row in h)
        useful = self.batch * self.hkv * self.q_tiles * TOPK
        visited = 0
        overlap = []
        for b in range(self.batch):
            for h in range(self.hkv):
                for m, (full, partial) in enumerate(zip(self.full[b][h], self.partial[b][h])):
                    valid_halves = min(self.q_stage, self.q_tiles - m * self.q_stage)
                    visited += valid_halves * (len(full) + len(partial))
                for m in range(0, self.q_tiles - 1, 2):
                    overlap.append(len(set(self.selected[b][h][m]) & set(self.selected[b][h][m + 1])))
        return {
            "q_stage": self.q_stage, "sparse_block_size": [BLOCK * self.q_stage, BLOCK],
            "full_block_count": nfull, "partial_block_count": npartial,
            "useful_q128_kv128_pairs_per_kv_head": useful,
            "union_schedule_pair_estimate": visited,
            "union_schedule_work_ratio": visited / useful,
            "work_ratio_note": "metadata-based estimate, not hardware instruction profiling or latency",
            "mean_adjacent_overlap_fraction": sum(overlap) / (TOPK * len(overlap)) if overlap else None,
            "block_lookup_bytes": self.batch * self.hkv * self.q_tiles * self.kv_tiles,
        }


def build_plan(selected, kv_tiles, q_stage):
    if q_stage not in (1, 2):
        raise ValueError("only q_stage=1/2 are in this experiment")
    batch, hkv, q_tiles = len(selected), len(selected[0]), len(selected[0][0])
    full, partial, allowed = [], [], []
    for b in range(batch):
        f_heads, p_heads, a_heads = [], [], []
        if len(selected[b]) != hkv:
            raise ValueError("ragged head shape")
        for h in range(hkv):
            if len(selected[b][h]) != q_tiles:
                raise ValueError("ragged query shape")
            sets = []
            for row in selected[b][h]:
                if len(row) != TOPK or len(set(row)) != TOPK or any(not 0 <= x < kv_tiles for x in row):
                    raise ValueError("lists must contain sixteen distinct in-range blocks")
                sets.append(set(row))
            fs, ps = [], []
            for m in range(0, q_tiles, q_stage):
                group = sets[m:m + q_stage]
                common, union = set.intersection(*group), set.union(*group)
                fs.append(sorted(common))
                ps.append(sorted(union - common))
            f_heads.append(fs)
            p_heads.append(ps)
            a_heads.append([[int(n in row) for n in range(kv_tiles)] for row in sets])
        full.append(f_heads)
        partial.append(p_heads)
        allowed.append(a_heads)
    plan = Plan(selected, kv_tiles, q_stage, full, partial, allowed)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Exhaustive block-grid check, independent of tensor layouts and CUDA."""
    for b in range(plan.batch):
        for h in range(plan.hkv):
            for q, row in enumerate(plan.selected[b][h]):
                m = q // plan.q_stage
                full, partial = set(plan.full[b][h][m]), set(plan.partial[b][h][m])
                if full & partial:
                    raise AssertionError("full and partial metadata overlap")
                actual = full | {n for n in partial if plan.allowed[b][h][q][n]}
                if actual != set(row):
                    raise AssertionError("adapter changed the mathematical mask")


def tensor_metadata(plan, *, hq, folded, device="cuda"):
    import torch
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    group = hq // plan.hkv
    if hq % plan.hkv or group != 16:
        raise ValueError("this initial FA4 benchmark deliberately uses G16 only")
    mblocks = math.ceil(plan.q_tiles / plan.q_stage)
    index_capacity = min(plan.kv_tiles, plan.q_stage * TOPK)

    def tensors(rows):
        counts = torch.tensor([[[len(row) for row in head] for head in b] for b in rows], dtype=torch.int32)
        padded = [[[[*row, *([0] * (index_capacity - len(row)))] for row in head] for head in b] for b in rows]
        indices = torch.tensor(padded, dtype=torch.int32)
        if folded:
            counts = counts.reshape(plan.batch * plan.hkv, 1, mblocks)
            indices = indices.reshape(plan.batch * plan.hkv, 1, mblocks, index_capacity)
        elif plan.hkv != 1:
            counts = counts.repeat_interleave(group, dim=1)
            indices = indices.repeat_interleave(group, dim=1)
        return counts.to(device).contiguous(), indices.to(device).contiguous()

    fc, fi = tensors(plan.full)
    pc, pi = tensors(plan.partial)
    lookup = torch.tensor(plan.allowed, dtype=torch.uint8)
    if folded:
        lookup = lookup.reshape(plan.batch * plan.hkv, 1, plan.q_tiles, plan.kv_tiles)
    metadata = BlockSparseTensorsTorch(pc, pi, fc, fi, block_size=(plan.q_stage * BLOCK, BLOCK))
    return metadata, lookup.to(device).contiguous()
