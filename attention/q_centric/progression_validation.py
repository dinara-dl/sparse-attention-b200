"""Independent FP32 checks for dense14 main p02/p03 and archived p01 control."""

import math
import random

try:
    from .progression import stage_functions
except ImportError:
    from progression import stage_functions


def make_inputs(shape, seed=17, scale=1.0):
    import torch
    b, hq, hkv, sq, skv = shape
    torch.manual_seed(seed)
    q = (torch.randn((b, hq, sq, 128), device="cuda") * scale).to(torch.bfloat16)
    k = (torch.randn((b, hkv, skv, 128), device="cuda") * scale).to(torch.bfloat16)
    return q, k, torch.randn_like(k)


def make_ids(b, hkv, qtiles, kvtiles, topk, seed, pattern="independent"):
    import torch
    if not 0 < topk <= kvtiles or pattern not in ("independent", "unsorted", "reverse", "paired"):
        raise ValueError("invalid TopK or pattern")
    rng = random.Random(seed)
    result = []
    for _ in range(b):
        heads = []
        for _ in range(hkv):
            rows = []
            for qt in range(qtiles):
                if pattern == "paired" and qt % 2:
                    row = list(rows[-1])
                else:
                    row = rng.sample(range(kvtiles), topk)
                    if pattern != "unsorted":
                        row.sort(reverse=pattern == "reverse")
                rows.append(row)
            heads.append(rows)
        result.append(heads)
    return torch.tensor(result, device="cuda", dtype=torch.int32)


def reference(q, k, v, ids):
    """Full FP32 output, without BF16-P rounding or the CUDA approximations."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    b, hq, sq, dim = q.shape
    hkv, group = k.shape[1], hq // k.shape[1]
    output = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    offsets = torch.arange(128, device=q.device)
    for batch in range(b):
        for head in range(hkv):
            for qt in range(sq // 128):
                positions = (ids[batch, head, qt].long()[:, None] * 128 + offsets).flatten()
                queries = q[batch, head*group:(head+1)*group, qt*128:(qt+1)*128].float()
                keys = k[batch, head].index_select(0, positions).float()
                values = v[batch, head].index_select(0, positions).float()
                output[batch, head*group:(head+1)*group, qt*128:(qt+1)*128] = (
                    torch.softmax(queries @ keys.T / math.sqrt(dim), dim=-1) @ values
                )
    return output


def check_output(actual, expected):
    import torch
    actual, expected = actual.float(), expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("non-finite output/reference")
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    delta = actual - expected
    return {"max_abs": delta.abs().max().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-20)).item(),
            "checked_output_elements": actual.numel(), "atol": 0.02, "rtol": 0.02}


def check_stage_correctness(version):
    import torch
    attention, attention_out = stage_functions(version)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [
        ((1, 2, 2, 256, 128), 1, "independent", 1.0),
        ((1, 2, 2, 512, 512), 2, "reverse", 0.25),
        ((2, 4, 2, 256, 512), 3, "unsorted", 1.0),
        ((1, 8, 2, 512, 2048), 16, "reverse", 1.0),
        ((1, 16, 2, 512, 4096), 16, "unsorted", 2.0),
        ((1, 32, 2, 256, 4096), 16, "independent", 4.0),
        ((1, 2, 2, 512, 4096), 32, "reverse", 1.0),
        ((2, 16, 16, 4096, 4096), 16, "independent", 1.0),
        ((2, 16, 1, 4096, 16384), 16, "independent", 1.0),
        ((2, 16, 1, 4096, 16384), 16, "paired", 1.0),
    ]
    if version == "p03":
        cases = [c for c in cases if c[0][1] != c[0][2]]
        cases += [((1, 4, 2, 128, 128), 1, "independent", 1.0),
                  ((1, 8, 2, 384, 512), 3, "reverse", 1.0),
                  ((1, 4, 2, 256, 4096), 32, "reverse", 1.0)]
    records = []
    for i, (shape, count, pattern, scale) in enumerate(cases):
        print(f"CHECK {version} shape={shape} topk={count} scale={scale}", flush=True)
        q, k, v = make_inputs(shape, 1700+i, scale)
        b, hq, hkv, sq, skv = shape
        ids = make_ids(b, hkv, sq//128, skv//128, count, 2300+i, pattern)
        expected = reference(q, k, v, ids)
        actual = attention(q, k, v, ids)
        stats = check_output(actual, expected)
        buffer = torch.full_like(q, float("nan"))
        attention_out(q, k, v, ids, buffer)
        torch.testing.assert_close(actual, buffer, rtol=0, atol=0)
        other = "p02" if version == "p03" else "p01" if version == "p02" else None
        if other is not None and sq % 256 == 0:
            control = stage_functions(other)[0](q, k, v, ids)
            torch.testing.assert_close(actual, control, rtol=0, atol=0)
            stats["cross_backend_exact"] = True
        records.append({"shape": list(shape), "topk": count, "pattern": pattern,
                        "scale": scale, "stats": stats, "out_api_exact": True})
    stream = torch.cuda.Stream()
    replay_checks = []
    with torch.cuda.stream(stream):
        q, k, v = make_inputs((2, 16, 2, 1024, 4096), 601)
        ids = make_ids(2, 2, 8, 32, 16, 601, "unsorted")
        output = torch.empty_like(q)
        for _ in range(3):
            attention_out(q, k, v, ids, output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        attention_out(q, k, v, ids, output)
        attention_out(q, k, v, ids, output)
    torch.cuda.synchronize()
    for seed in (607, 613, 617):
        with torch.cuda.stream(stream):
            q.mul_(-0.75)
            v.add_(0.0625)
            ids.copy_(make_ids(2, 2, 8, 32, 16, seed, "unsorted"))
            output.fill_(float("nan"))
            graph.replay()
            replay_checks.append(check_output(output, reference(q, k, v, ids)))
        stream.synchronize()
    rejected = []
    for name, args in (
        ("zero_kv_heads", (q, k[:, :0], v[:, :0], ids[:, :0])),
        ("empty_ids", (q, k, v, ids[..., :0])),
        ("fp32_q", (q.float(), k, v, ids)),
        ("requires_grad", (q.detach().requires_grad_(), k, v, ids)),
        ("per_q_head_ids", (q, k, v, ids.repeat_interleave(8, dim=1))),
    ):
        try:
            attention(*args)
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(f"accepted invalid {name}")
    for name, out in (("output_alias", q), ("output_shape", output[:, :, :128])):
        try:
            attention_out(q, k, v, ids, out)
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(f"accepted invalid {name}")
    torch.cuda.synchronize()
    result = {"kernel": version, "passed": True, "cases": records,
              "dynamic_graph_checks": replay_checks, "rejected_inputs": rejected}
    for count in (1, 3, 16):
        result[f"topk{count}_max_abs"] = max(r["stats"]["max_abs"] for r in records if r["topk"] == count)
    result["all_tiles_max_abs"] = max(r["stats"]["max_abs"] for r in records if r["topk"]*128 == r["shape"][4])
    result["persistent_two_wave_max_abs"] = max(r["stats"]["max_abs"] for r in records if r["shape"][3] == 4096)
    return result
