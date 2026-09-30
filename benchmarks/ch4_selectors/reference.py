"""Independent PyTorch scoring/oracle and explicitly synthetic input families."""
import math
import torch


def make_inputs(batch, sq, skv, family, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    def rand(shape):
        return torch.randn(shape, device="cuda", generator=gen)
    if family == "gaussian":
        q, k = rand((batch, 16, sq, 128)), rand((batch, 1, skv, 128))
    elif family == "cancellation_stress":
        qt, nt = sq // 128, skv // 128
        centers = rand((batch, nt, 128))
        centers = centers / centers.norm(dim=-1, keepdim=True) * math.sqrt(128)
        order = torch.rand((batch, qt, nt), device="cuda", generator=gen).argsort(dim=-1)
        targets = order[..., :4][:, :, torch.arange(16, device="cuda") % 4].permute(0, 2, 1)
        qc = centers[torch.arange(batch, device="cuda")[:, None, None], targets]
        signs = torch.where(torch.arange(128, device="cuda") < 64, 1.0, -1.0)[None, None, None, :, None]
        q = (qc[..., None, :] * signs + .1 * rand((batch, 16, qt, 128, 128))).reshape(batch, 16, sq, 128)
        k = (centers[:, None, :, None, :] * signs + .1 * rand((batch, 1, nt, 128, 128))).reshape(batch, 1, skv, 128)
    else:
        raise ValueError(family)
    v = rand((batch, 1, skv, 128))
    return tuple(x.to(torch.bfloat16) for x in (q, k, v))


def summary_reference(q, k, r):
    b, h, sq, _ = q.shape
    qt, nt = sq // 128, k.shape[2] // 128
    # FP32 pooling followed by the SAME declared BF16 descriptor rounding.
    qd = q.float().reshape(b, h, qt, r, 128 // r, 128).mean(dim=-2).to(torch.bfloat16)
    kd = k.float().reshape(b, 1, nt, r, 128 // r, 128).mean(dim=-2).to(torch.bfloat16)
    raw = torch.bmm(qd.float().reshape(b, h * qt * r, 128),
                    kd.float().reshape(b, nt * r, 128).transpose(1, 2)) / math.sqrt(128)
    scores = raw.reshape(b, h, qt, r, nt, r).amax(dim=(1, 3, 5))[:, None]
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :16].to(torch.int32)
    return qd, kd, scores, ids


def oracle(q, k, v):
    """Full FP32 dense output and per-query/head KV-block probability mass."""
    b, h, sq, _ = q.shape
    nt = k.shape[2] // 128
    masses = torch.empty((b, h, sq, nt), device=q.device, dtype=torch.float32)
    output = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for bi in range(b):
        keys, values = k[bi, 0].float(), v[bi, 0].float()
        for start in range(0, sq, 128):
            prob = torch.softmax(q[bi, :, start:start+128].float() @ keys.T / math.sqrt(128), dim=-1)
            masses[bi, :, start:start+128] = prob.reshape(h, 128, nt, 128).sum(-1)
            output[bi, :, start:start+128] = prob @ values
    tile_mass = masses.reshape(b, h, sq // 128, 128, nt).mean(dim=(1, 3))[:, None]
    ids = tile_mass.argsort(dim=-1, descending=True, stable=True)[..., :16].to(torch.int32)
    return {"masses": masses, "output": output, "tile_mass": tile_mass, "ids": ids}


def captured_mass(ids, dense):
    mass = dense["masses"]
    b, h, sq, nt = mass.shape
    selected = ids[:, :, :, None, :].expand(b, h, sq // 128, 128, 16).long()
    return mass.reshape(b, h, sq // 128, 128, nt).gather(-1, selected).sum(-1)


def quality(ids, sparse_pieces, dense):
    capture = captured_mass(ids, dense)
    best_capture = captured_mass(dense["ids"], dense)
    oracle_ids = dense["ids"]
    recall = (ids[..., :, None] == oracle_ids[..., None, :]).any(-1).float().mean()
    mse_sum, norm_sum, count, max_abs = 0.0, 0.0, 0, 0.0
    for b, heads, rows, sparse in sparse_pieces:
        target = dense["output"][b, heads].index_select(1, rows)
        diff = sparse - target
        mse_sum += diff.square().sum().item()
        norm_sum += target.square().sum().item()
        count += diff.numel()
        max_abs = max(max_abs, diff.abs().max().item())
    return {"captured_mass_mean": capture.mean().item(), "captured_mass_min": capture.min().item(),
            "captured_mass_p01": torch.quantile(capture.flatten(), .01).item(),
            "captured_mass_p05": torch.quantile(capture.flatten(), .05).item(),
            "captured_mass_p50": capture.median().item(), "oracle_tile_mass_mean": best_capture.mean().item(),
            "oracle_mass_efficiency": capture.mean().item() / best_capture.mean().item(),
            "recall_vs_tile_mass_oracle": recall.item(), "output_rel_l2": math.sqrt(mse_sum / max(norm_sum, 1e-30)),
            "output_rmse": math.sqrt(mse_sum / count), "output_max_abs": max_abs,
            "quality_reference": "FP32 selected attention vs FP32 dense; numerical BF16 kernel error excluded",
            "oracle_note": "oracle optimizes average captured mass for Q128/GQA masks, not output error or model loss"}
