"""Independent FP32 reference for the fixed non-causal NVFP4 input contract."""


def dequantize_nvfp4(packed, scales):
    import torch

    # E2M1: low nibble is the first value, high nibble is the second.
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], 128
    ).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              device=packed.device, dtype=torch.float32)
    values = magnitudes[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return values * scales.float().repeat_interleave(16, dim=-1)


def fp4_scores_reference(inputs):
    import torch

    q = dequantize_nvfp4(inputs["q"], inputs["q_scale"])
    k = dequantize_nvfp4(inputs["k"], inputs["k_scale"])
    batch, hkv = inputs["batch_size"], inputs["hkv"]
    seq_q, seq_kv = inputs["len_q"], inputs["len_kv"]
    kv_tiles = seq_kv // 128
    scores = torch.empty((hkv, kv_tiles, batch * seq_q), device=q.device,
                          dtype=torch.float32)
    for b in range(batch):
        keys = k[b * kv_tiles : (b + 1) * kv_tiles, 0].reshape(seq_kv, 128)
        start = b * seq_q
        for h in range(hkv):
            products = q[start : start + seq_q, h] @ keys.T
            scores[h, :, start : start + seq_q] = (
                products.reshape(seq_q, kv_tiles, 128).amax(dim=-1).T
            )
    return scores
