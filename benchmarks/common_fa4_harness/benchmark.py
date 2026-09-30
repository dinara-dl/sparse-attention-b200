"""Stage 1 only: prepared IDs -> BF16 output. No indexer or model inference."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from metadata import build_plan, make_lists

CURRENT = {"records": [], "checks": []}
ATOL, RTOL = 0.02, 0.02


def load_ours(here):
    from torch.utils.cpp_extension import load
    base = here / "baseline"
    library = load(
        name="fa4_compare_frozen_s10",
        sources=[str(base / "native_gqa_attention.cpp"), str(base / "10_native_gqa_block_sparse.cu")],
        extra_include_paths=[str(base)], extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v", "-gencode=arch=compute_100a,code=sm_100a"],
        extra_ldflags=["-lcuda"], is_python_module=False, verbose=True,
    )
    resource = subprocess.run(["cuobjdump", "--dump-resource-usage", str(library)], capture_output=True, text=True, check=False)
    CURRENT["our_resource_usage"] = resource.stdout
    return torch.ops.xomtention_native_gqa.attention_out


def gpu_state():
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pstate", "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def error_stats(actual, reference):
    actual, reference = actual.float(), reference.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError("non-finite output/reference")
    diff = actual - reference
    result = {
        "max_abs": diff.abs().max().item(), "rmse": diff.square().mean().sqrt().item(),
        "relative_l2": (diff.norm() / reference.norm().clamp_min(1e-20)).item(),
        "atol": ATOL, "rtol": RTOL,
    }
    torch.testing.assert_close(actual, reference, atol=ATOL, rtol=RTOL)
    return result


def reference(q, k, v, ids, *, sample=False, dense=False):
    """Independent FP32 selected-token QK/softmax/PV; no kernel approximations."""
    batch, hq, sq, dim = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    result = []
    offsets = torch.arange(128, device=q.device)
    bvalues = sorted({0, batch - 1}) if sample else range(batch)
    qtiles = sorted({0, 1, sq // 128 - 1}) if sample else range(sq // 128)
    for b in bvalues:
        for h in range(hkv):
            heads = [h * group, (h + 1) * group - 1] if sample else list(range(h * group, (h + 1) * group))
            for qt in qtiles:
                if sample:
                    rows = torch.tensor([qt * 128, qt * 128 + 31, qt * 128 + 127], device=q.device)
                else:
                    rows = torch.arange(qt * 128, (qt + 1) * 128, device=q.device)
                positions = torch.arange(k.shape[2], device=q.device) if dense else (ids[b, h, qt].long()[:, None] * 128 + offsets).flatten()
                qrows = q[b, heads].index_select(1, rows).float()
                kvals = k[b, h].index_select(0, positions).float()
                vvals = v[b, h].index_select(0, positions).float()
                expected = torch.softmax(qrows @ kvals.T / math.sqrt(dim), dim=-1) @ vvals
                result.append((b, heads, rows, expected))
    return result


def check_reference(output, pieces):
    actual = torch.cat([output[b, heads].index_select(1, rows).reshape(-1) for b, heads, rows, _ in pieces])
    expected = torch.cat([value.reshape(-1) for _, _, _, value in pieces])
    stats = error_stats(actual, expected)
    stats["checked_output_elements"] = actual.numel()
    return stats


def capture(fn, calls=8):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph, calls


def time_graph(captured, warmup_ms, measure_ms):
    graph, calls = captured

    def timed(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    estimate = max(timed(8) / 8, 0.001)
    timed(max(1, math.ceil(warmup_ms / estimate)))
    repeats = max(8, math.ceil(measure_ms / estimate))
    return timed(repeats) / (repeats * calls)


def make_case(ours, shape, pattern, seed, scale=1.0, reverse=False):
    batch, hq, hkv, sq, skv = shape
    torch.manual_seed(seed)
    q = (torch.randn((batch, hq, sq, 128), device="cuda") * scale).to(torch.bfloat16)
    k = (torch.randn((batch, hkv, skv, 128), device="cuda") * scale).to(torch.bfloat16)
    v = torch.randn_like(k)
    lists = make_lists(batch, hkv, sq // 128, skv // 128, pattern, seed + 1000)
    if reverse:
        lists = [[[row[::-1] for row in h] for h in b] for b in lists]
    ids = torch.tensor(lists, device="cuda", dtype=torch.int32)
    out = torch.full_like(q, float("nan"))

    def own():
        ours(q, k, v, ids, out)

    plans = {stage: build_plan(lists, skv // 128, stage) for stage in (1, 2)}
    return q, k, v, ids, out, own, plans


def run_case(ours, shape, pattern, seed, *, scale=1.0, reverse=False,
             timing=False, full_reference=False, native_control=False,
             warmup_ms=250, measure_ms=1000, rounds=3):
    from fa4_adapter import Fa4Backend
    started = time.monotonic()
    q, k, v, ids, out, own, plans = make_case(ours, shape, pattern, seed, scale, reverse)
    case = {"shape": dict(zip(("batch", "hq", "hkv", "sq", "skv"), shape)),
            "pattern": pattern, "seed": seed, "qk_scale": scale, "reverse_ids": reverse,
            "topk": 16, "block": 128, "dim": 128, "dtype": "BF16", "causal": False,
            "selected_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
    print("CASE_START " + json.dumps(case), flush=True)
    own()
    expected = reference(q, k, v, ids, sample=not full_reference)
    checks = {"ours_vs_fp32": check_reference(out, expected)}
    funcs, backends = {"ours": own}, {}
    for stage, pack in ((2, True), (2, False), (1, True), (1, False)):
        name = f"fa4_q{stage}_{'pack' if pack else 'unpack'}"
        backend = Fa4Backend(q, k, v, plan=plans[stage], q_stage=stage, pack=pack, folded=True)
        backend.run(observe=True)
        logical = backend.logical_output()
        checks[name + "_vs_fp32"] = check_reference(logical, expected)
        checks[name + "_vs_ours_full"] = error_stats(logical, out)
        backends[name], funcs[name] = backend, backend.run
        print(f"BACKEND_OK {name} elapsed_s={time.monotonic() - started:.2f} config={backend.description()}", flush=True)
    if native_control and shape[2] > 1:
        backend = Fa4Backend(q, k, v, plan=plans[2], q_stage=2, pack=True, folded=False)
        backend.run(observe=True)
        checks["fa4_native_dispatch_vs_fp32"] = check_reference(backend.logical_output(), expected)
        checks["fa4_native_dispatch_vs_ours_full"] = error_stats(backend.logical_output(), out)
        backends["fa4_native_dispatch"], funcs["fa4_native_dispatch"] = backend, backend.run
    dense = Fa4Backend(q, k, v, folded=False, pack=True)
    dense.run(observe=True)
    dense_expected = expected if shape[-1] == 2048 else reference(q, k, v, ids, sample=True, dense=True)
    checks["fa4_dense_vs_fp32"] = check_reference(dense.logical_output(), dense_expected)
    if shape[-1] == 2048:
        checks["all_blocks_sparse_vs_dense"] = error_stats(out, dense.logical_output())
    backends["fa4_dense"], funcs["fa4_dense"] = dense, dense.run
    torch.cuda.synchronize()
    case.update(correctness=checks, configs={name: b.description() for name, b in backends.items()},
                fp32_reference="all output rows" if full_reference else "sampled rows across both batches, every KV group, first/second/last Q tiles; full backend-to-backend comparison",
                our_work_items=shape[0] * (shape[1] // 2) * (shape[3] // 128),
                gpu_sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    if timing:
        graphs = {name: capture(fn) for name, fn in funcs.items()}
        samples = {name: [] for name in funcs}
        base_order = list(funcs)
        orders = []
        case["gpu_before"] = gpu_state()
        for rnd in range(rounds):
            order = base_order[rnd:] + base_order[:rnd]
            if rnd % 2:
                order = list(reversed(order))
            orders.append(order)
            for name in order:
                value = time_graph(graphs[name], warmup_ms, measure_ms)
                samples[name].append(value)
                print(f"TIME shape={shape} pattern={pattern} seed={seed} round={rnd} backend={name} ms={value:.6f}", flush=True)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        useful = 4 * shape[0] * shape[1] * shape[3] * 2048 * 128
        dense_flops = 4 * shape[0] * shape[1] * shape[3] * shape[4] * 128
        sparse_names = [name for name in medians if name.startswith("fa4_") and name != "fa4_dense"]
        best = min(sparse_names, key=medians.get)
        case.update(samples_ms=samples, median_ms=medians, measurement_orders=orders,
                    sparse_qk_pv_flops=useful, dense_qk_pv_flops=dense_flops,
                    useful_tflops={name: useful / (value * 1e9) for name, value in medians.items() if name != "fa4_dense"},
                    dense_tflops=dense_flops / (medians["fa4_dense"] * 1e9),
                    best_verified_fa4=best, speedup_vs_best_fa4=medians[best] / medians["ours"],
                    speedup_vs_dense=medians["fa4_dense"] / medians["ours"], gpu_after=gpu_state())
    case["elapsed_seconds"] = time.monotonic() - started
    print("CASE_RESULT " + json.dumps(case), flush=True)
    return case


def run(here, mode, warmup_ms=250, measure_ms=1000, rounds=3):
    global CURRENT
    CURRENT = {"records": [], "checks": [], "mode": mode}
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    CURRENT.update(device=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                   cuda_version=torch.version.cuda, gpu_start=gpu_state(),
                   versions={p: importlib.metadata.version(p) for p in (
                       "nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi", "torch-c-dlpack-ext", "triton")})
    ours = load_ours(here)
    print(f"OURS_BUILD_DONE elapsed_s={time.monotonic() - started:.2f}", flush=True)
    if mode == "correctness":
        configs = [
            ((1, 16, 1, 384, 2048), "all", 17, 1.0, False),
            ((1, 16, 1, 384, 4096), "disjoint", 23, 1.0, False),
            ((1, 64, 4, 384, 4096), "independent", 29, 1.0, True),
            ((1, 16, 1, 512, 4096), "independent", 31, 4.0, True),
            ((2, 16, 1, 4096, 4096), "independent", 37, 0.25, False),
        ]
        for shape, pattern, seed, scale, reverse in configs:
            if time.monotonic() - started > 370:
                CURRENT.update(passed=False, incomplete=True, reason="split correctness to respect eight-minute timeout")
                return CURRENT
            record = run_case(ours, shape, pattern, seed, scale=scale, reverse=reverse,
                              full_reference=True, native_control=shape[2] > 1)
            CURRENT["checks"].append(record)
    else:
        if mode == "core_short":
            profiles = [(2, 16, 1, 4096, skv) for skv in (2048, 4096)]
        elif mode == "core_long":
            profiles = [(2, 16, 1, 4096, skv) for skv in (8192, 16384)]
        elif mode == "gqa_groups":
            profiles = [(2, 64, 4, 4096, skv) for skv in (4096, 16384)]
        elif mode == "hot_confirmation":
            profiles = [(2, 16, 1, 4096, 16384), (2, 64, 4, 4096, 16384)]
        else:
            raise ValueError(mode)
        for shape in profiles:
            patterns = ["all"] if shape[-1] == 2048 else (["hot"] if mode == "hot_confirmation" else ["paired", "independent"])
            seeds = [17, 29] if mode != "gqa_groups" else [17]
            for pattern in patterns:
                for seed in seeds:
                    if time.monotonic() - started > 365:
                        CURRENT.update(passed=False, incomplete=True, reason="split benchmark to respect eight-minute timeout")
                        return CURRENT
                    record = run_case(ours, shape, pattern, seed, timing=True,
                                      native_control=shape[2] > 1, warmup_ms=warmup_ms,
                                      measure_ms=measure_ms, rounds=rounds)
                    CURRENT["records"].append(record)
    CURRENT.update(passed=True, elapsed_seconds=time.monotonic() - started,
                   gpu_end=gpu_state(), timer={"kind": "CUDA events around CUDA Graph replay",
                   "calls_per_graph": 8, "warmup_ms_per_backend_per_round": warmup_ms,
                   "measure_ms_per_backend_per_round": measure_ms, "rounds": rounds,
                   "cache_regime": "steady-state same-buffer graph replay; identical policy for all backends"},
                   timed_region="prepared selected IDs / FA4 metadata and preferred input layout -> preallocated final BF16 O",
                   excluded=["indexer", "metadata construction", "layout conversion", "input generation", "compilation", "reference checks"],
                   first_party_cuda_changed=False, reference_cuda_changed=False,
                   sanitizer_run=False, model_quality_claim=False)
    return CURRENT
