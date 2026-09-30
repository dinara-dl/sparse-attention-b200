"""Independent PyTorch reference for query-tile reduction and TopK16."""


def tile_topk_reference(scores, *, batch_size: int, len_q: int):
    import torch

    heads, kv_tiles, _ = scores.shape
    # Expose the 128-query axis, then reduce it to one score per KV block.
    tile_scores = (
        scores.reshape(heads, kv_tiles, batch_size, len_q // 128, 128)
        .amax(dim=-1)
        .permute(2, 0, 3, 1)
    )
    # Unlike topk's unspecified tie order, stable argsort keeps smaller IDs first.
    return torch.argsort(
        tile_scores, dim=-1, descending=True, stable=True
    )[..., :16].to(torch.int32).contiguous()
