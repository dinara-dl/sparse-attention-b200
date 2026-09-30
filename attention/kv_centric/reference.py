"""Independent FP32 oracle: one global softmax, no reverse metadata."""
import math
import torch
from metadata import BLOCK, TOPK, validate_selected, validate_plan


@torch.no_grad()
def attention_reference(q, k, v, selected, query_chunk=32):
    validate_selected(selected, k.shape[2] // BLOCK)
    bsz, hq, sq, dim = q.shape
    hkv, skv = k.shape[1:3]
    group_size = hq // hkv
    if tuple(selected.shape) != (bsz, hkv, sq, TOPK):
        raise ValueError("selected shape does not match Q/K")
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    for b in range(bsz):
        for h in range(hkv):
            kh, vh = k[b, h].float(), v[b, h].float()
            heads = slice(h * group_size, (h + 1) * group_size)
            for first in range(0, sq, query_chunk):
                end = min(first + query_chunk, sq)
                block_mask = torch.zeros((end - first, skv // BLOCK),
                                         dtype=torch.bool, device=q.device)
                block_mask.scatter_(1, selected[b, h, first:end].to(q.device).long(), True)
                mask = block_mask.repeat_interleave(BLOCK, dim=-1)
                scores = (q[b, heads, first:end].float() @ kh.T) / math.sqrt(dim)
                scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)
                out[b, heads, first:end] = torch.softmax(scores, dim=-1) @ vh
                lse[b, heads, first:end] = torch.logsumexp(scores, dim=-1)
    return out, lse


def combine_reference(partial_o, partial_lse):
    # Shapes [...,16,D] and [...,16]; normalization is over selected blocks.
    lse = torch.logsumexp(partial_lse.double(), dim=-1)
    weights = torch.exp(partial_lse.double() - lse.unsqueeze(-1))
    out = (partial_o.double() * weights.unsqueeze(-1)).sum(dim=-2)
    return out.float(), lse.float()


@torch.no_grad()
def simulate_kv_path(q, k, v, selected, plan):
    """CPU/torch audit of scheduled ownership and rank-addressed partial slots."""
    validate_plan(plan, selected)
    bsz, hq, sq, dim = q.shape
    partial = torch.full((bsz, hq, sq, TOPK, dim), torch.nan,
                         dtype=torch.float32, device=q.device)
    lses = torch.full((bsz, hq, sq, TOPK), torch.nan,
                      dtype=torch.float32, device=q.device)
    g = plan.group_size
    for group, block, start, count in plan.tasks.tolist():
        b, h = divmod(group, plan.hkv)
        qids = plan.query_ids[start:start + count].long().to(q.device)
        ranks = plan.ranks[start:start + count].long().to(q.device)
        heads = torch.arange(h * g, (h + 1) * g, device=q.device)
        queries = q[b, heads[:, None], qids[None, :]].float()
        kb = k[b, h, block * BLOCK:(block + 1) * BLOCK].float()
        vb = v[b, h, block * BLOCK:(block + 1) * BLOCK].float()
        scores = queries @ kb.T / math.sqrt(dim)
        part = torch.softmax(scores, dim=-1) @ vb
        partial[b, heads[:, None], qids[None, :], ranks[None, :]] = part
        lses[b, heads[:, None], qids[None, :], ranks[None, :]] = torch.logsumexp(scores, -1)
    return (*combine_reference(partial, lses), partial, lses)
