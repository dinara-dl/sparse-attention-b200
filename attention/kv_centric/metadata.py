"""Deliberately simple CPU metadata for the first KV-owned CUDA backend."""
from dataclasses import dataclass
import torch

BLOCK = 128
TOPK = 16
GROUPS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class Plan:
    batch: int
    hkv: int
    sq: int
    kv_blocks: int
    group_size: int
    row_ptr: torch.Tensor
    query_ids: torch.Tensor
    ranks: torch.Tensor
    tasks: torch.Tensor  # [task, (flat batch/head, block, edge start, edge count)]


def validate_selected(selected, kv_blocks):
    if not isinstance(selected, torch.Tensor):
        raise TypeError("selected must be a CPU int32 tensor")
    if selected.device.type != "cpu" or selected.dtype != torch.int32:
        raise ValueError("selected must be CPU int32; GPU metadata is a later stage")
    if selected.ndim != 4 or selected.shape[-1] != TOPK:
        raise ValueError("selected must have shape [B,Hkv,Sq,16], not Q tiles")
    if not selected.is_contiguous() or any(x <= 0 for x in selected.shape):
        raise ValueError("selected must be nonempty and contiguous")
    if not isinstance(kv_blocks, int) or kv_blocks < TOPK:
        raise ValueError("at least 16 complete KV blocks are required")
    if selected.numel() >= 2**31 or selected.shape[0] * selected.shape[1] * kv_blocks >= 2**31:
        raise ValueError("v0 metadata is limited to signed int32 offsets")
    ordered = selected.sort(dim=-1).values
    if ordered[..., 0].min().item() < 0 or ordered[..., -1].max().item() >= kv_blocks:
        raise ValueError("selected block ID is out of range")
    if torch.any(ordered[..., 1:] == ordered[..., :-1]).item():
        raise ValueError("each query/group must select 16 unique blocks")


def build_plan(selected, kv_blocks, group_size):
    validate_selected(selected, kv_blocks)
    if group_size not in GROUPS:
        raise ValueError(f"group_size must be one of {GROUPS}")
    batch, hkv, sq, _ = selected.shape
    rows = [[] for _ in range(batch * hkv * kv_blocks)]
    for group_id, lists in enumerate(selected.reshape(-1, sq, TOPK).tolist()):
        for query, ids in enumerate(lists):
            for rank, block in enumerate(ids):
                rows[group_id * kv_blocks + block].append((query, rank))
    row_ptr, qids, ranks, tasks = [0], [], [], []
    query_chunk = BLOCK // group_size
    for row, edges in enumerate(rows):
        start = len(qids)
        qids.extend(q for q, _ in edges)
        ranks.extend(rank for _, rank in edges)
        for offset in range(0, len(edges), query_chunk):
            tasks.append((row // kv_blocks, row % kv_blocks, start + offset,
                          min(query_chunk, len(edges) - offset)))
        row_ptr.append(len(qids))
    plan = Plan(batch, hkv, sq, kv_blocks, group_size,
                torch.tensor(row_ptr, dtype=torch.int32),
                torch.tensor(qids, dtype=torch.int32),
                torch.tensor(ranks, dtype=torch.int32),
                torch.tensor(tasks, dtype=torch.int32).reshape(-1, 4))
    validate_plan(plan, selected)
    return plan


def validate_plan(plan, selected):
    """Audit exact coverage independently of the builder's append loops."""
    validate_selected(selected, plan.kv_blocks)
    if tuple(selected.shape) != (plan.batch, plan.hkv, plan.sq, TOPK):
        raise ValueError("plan does not match selected shape")
    if plan.group_size not in GROUPS:
        raise ValueError("invalid plan GQA ratio")
    tensors = (plan.row_ptr, plan.query_ids, plan.ranks, plan.tasks)
    if any(t.device.type != "cpu" or t.dtype != torch.int32 or not t.is_contiguous()
           for t in tensors):
        raise ValueError("all plan tensors must be contiguous CPU int32")
    edges = selected.numel()
    nrows = plan.batch * plan.hkv * plan.kv_blocks
    if (plan.row_ptr.shape != (nrows + 1,) or plan.query_ids.shape != (edges,)
            or plan.ranks.shape != (edges,) or plan.tasks.ndim != 2
            or plan.tasks.shape[1] != 4):
        raise ValueError("invalid CSR/task dimensions")
    ptr, qids, ranks = plan.row_ptr.tolist(), plan.query_ids.tolist(), plan.ranks.tolist()
    if ptr[0] != 0 or ptr[-1] != edges or any(a > b for a, b in zip(ptr, ptr[1:])):
        raise ValueError("CSR row_ptr must be monotone and cover every edge")
    selected_flat = selected.reshape(-1, plan.sq, TOPK).tolist()
    seen = set()
    for row in range(nrows):
        group, block = divmod(row, plan.kv_blocks)
        for edge in range(ptr[row], ptr[row + 1]):
            q, rank = qids[edge], ranks[edge]
            if not (0 <= q < plan.sq and 0 <= rank < TOPK):
                raise ValueError("query/rank outside bounds")
            key = (group, q, rank)
            if key in seen or selected_flat[group][q][rank] != block:
                raise ValueError("duplicate or incorrectly reversed edge")
            seen.add(key)
    if len(seen) != edges:
        raise ValueError("CSR omitted a selected edge")
    visits = [0] * edges
    for group, block, start, count in plan.tasks.tolist():
        if not (0 <= group < plan.batch * plan.hkv and 0 <= block < plan.kv_blocks):
            raise ValueError("task group/block outside bounds")
        row = group * plan.kv_blocks + block
        if not (0 < count <= BLOCK // plan.group_size
                and ptr[row] <= start < start + count <= ptr[row + 1]):
            raise ValueError("task crosses CSR row or MMA capacity")
        for edge in range(start, start + count):
            visits[edge] += 1
    if any(count != 1 for count in visits):
        raise ValueError("tasks must visit every CSR edge exactly once")


def make_selected(batch, hkv, sq, kv_blocks, kind, seed=1):
    """Synthetic selectors for testing only; no model-quality interpretation."""
    if kv_blocks < TOPK:
        raise ValueError("need at least 16 blocks")
    gen = torch.Generator().manual_seed(seed)
    result = torch.empty((batch, hkv, sq, TOPK), dtype=torch.int32)
    for b in range(batch):
        for h in range(hkv):
            shared = torch.randperm(kv_blocks, generator=gen)[:TOPK]
            for q in range(sq):
                if kind == "tile_shared":
                    if q % BLOCK == 0:
                        shared = torch.randperm(kv_blocks, generator=gen)[:TOPK]
                    ids = shared
                elif kind == "identical":
                    ids = shared
                elif kind == "independent":
                    ids = torch.randperm(kv_blocks, generator=gen)[:TOPK]
                elif kind == "overlap":
                    tail = torch.randperm(kv_blocks - 8, generator=gen)[:8] + 8
                    ids = torch.cat((torch.arange(8), tail))
                elif kind == "hot":
                    tail = torch.randperm(kv_blocks - 1, generator=gen)[:15] + 1
                    ids = torch.cat((torch.zeros(1, dtype=torch.int64), tail))
                elif kind == "all_blocks" and kv_blocks == TOPK:
                    ids = torch.randperm(TOPK, generator=gen)
                else:
                    raise ValueError(f"unknown/incompatible mask kind: {kind}")
                result[b, h, q] = ids.to(torch.int32)
    return result
