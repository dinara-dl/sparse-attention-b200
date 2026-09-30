"""Current main p03; reuse the historical exact-mask adapters, not its CUDA."""

import hashlib
import importlib.metadata
import json
import statistics
import time

import torch

import benchmark as base
from controls import ControlledBackend
from progression import stage_functions
from protocol import (CALLS_PER_GRAPH, MEASURE_MS, PROTOCOL, ROUNDS, WARMUP_MS,
                      measurement_order, performance_cases)

CURRENT = {}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def one_case(ours, shape, pattern, seed, *, timing, scale=1.0, reverse=False):
    started = time.monotonic()
    q, k, v, ids, out, own, plans = base.make_case(ours, shape, pattern, seed, scale, reverse)
    record = {
        "shape": dict(zip(("batch", "hq", "hkv", "sq", "skv"), shape)),
        "pattern": pattern, "seed": seed, "qk_scale": scale, "reverse_ids": reverse,
        "topk": 16, "block_size": 128, "dim": 128, "dtype": "BF16", "causal": False,
        "selected_sha256": tensor_sha(ids),
        "input_sha256": {name: tensor_sha(value) for name, value in (("q", q), ("k", k), ("v", v))},
        "reference": "full independent FP32 QK/softmax/PV; TF32 disabled",
        "our_work_items": shape[0] * (shape[1] // 2) * (shape[3] // 128),
        "gpu_sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    }
    print("CASE_START " + json.dumps(record), flush=True)
    # Full references on the actual timed shapes, not sampled rows.
    sparse_ref = base.reference(q, k, v, ids)
    dense_ref = sparse_ref if shape[-1] == 2048 else base.reference(q, k, v, ids, dense=True)
    own()
    checks = {"ours_vs_fp32": base.check_reference(out, sparse_ref)}
    backends = {}
    for stage, pack in ((2, True), (2, False), (1, True), (1, False)):
        name = f"fa4_q{stage}_{'pack' if pack else 'unpack'}"
        backends[name] = ControlledBackend(q, k, v, plan=plans[stage], q_stage=stage,
                                          pack=pack, folded=True, vector=stage == 2)
    for pack, force in ((True, False), (False, False), (True, True), (False, True)):
        name = f"fa4_dense_{'pack' if pack else 'unpack'}_{'1cta' if force else 'autocta'}"
        backends[name] = ControlledBackend(q, k, v, pack=pack, folded=False, force_1cta=force)
    for name, backend in backends.items():
        backend.out.fill_(float("nan"))
        backend.run(observe=True)
        expected = dense_ref if backend.is_dense else sparse_ref
        checks[name + "_vs_fp32"] = base.check_reference(backend.logical_output(), expected)
        if not backend.is_dense or shape[-1] == 2048:
            checks[name + "_vs_ours"] = base.error_stats(backend.logical_output(), out)
        print(f"CHECK_OK {name} elapsed_s={time.monotonic()-started:.2f}", flush=True)
    record.update(correctness=checks, configs={name: b.description() for name, b in backends.items()})
    funcs = {"ours": own, **{name: b.run for name, b in backends.items()}}
    graphs = {name: base.capture(fn, calls=CALLS_PER_GRAPH) for name, fn in funcs.items()}
    if timing:
        samples = {name: [] for name in funcs}
        orders = []
        record["gpu_before"] = base.gpu_state()
        for rnd in range(ROUNDS):
            order = measurement_order(list(funcs), seed, rnd)
            orders.append(order)
            for name in order:
                ms = base.time_graph(graphs[name], WARMUP_MS, MEASURE_MS)
                samples[name].append(ms)
                print(f"TIME skv={shape[-1]} pattern={pattern} seed={seed} round={rnd} {name}={ms:.6f}ms", flush=True)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        best_sparse = min((name for name in medians if name.startswith("fa4_q")), key=medians.get)
        best_dense = min((name for name in medians if name.startswith("fa4_dense")), key=medians.get)
        useful_flops = 4 * shape[0] * shape[1] * shape[3] * 2048 * 128
        record.update(
            samples_ms=samples, median_ms=medians, measurement_orders=orders,
            best_fa4_sparse=best_sparse, best_fa4_dense=best_dense,
            speedup_vs_best_fa4_sparse=medians[best_sparse] / medians["ours"],
            speedup_vs_best_fa4_dense=medians[best_dense] / medians["ours"],
            sparse_qk_pv_flops=useful_flops,
            dense_qk_pv_flops=4 * shape[0] * shape[1] * shape[3] * shape[4] * 128,
            our_useful_tflops=useful_flops / (medians["ours"] * 1e9),
            gpu_after=base.gpu_state(),
        )
    # Refill every destination to verify that graph replay writes all outputs.
    post_checks = {}
    for name, (graph, _) in graphs.items():
        destination = out if name == "ours" else backends[name].out
        destination.fill_(float("nan"))
        graph.replay()
        logical = out if name == "ours" else backends[name].logical_output()
        expected = dense_ref if name.startswith("fa4_dense") else sparse_ref
        post_checks[name] = base.check_reference(logical, expected)
    record.update(post_graph_correctness=post_checks, elapsed_seconds=time.monotonic()-started)
    print(f"CASE_DONE skv={shape[-1]} pattern={pattern} seed={seed} elapsed_s={record['elapsed_seconds']:.2f}", flush=True)
    return record


def run(mode, skv):
    global CURRENT
    started = time.monotonic()
    CURRENT = {
        "passed": False, "mode": mode, "skv": skv, "checks": [], "records": [],
        "protocol": PROTOCOL, "our_version": "p03", "our_source": "sparse/02_gqa_block_sparse.cu",
        "device": torch.cuda.get_device_name(0), "gpu_start": base.gpu_state(),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "versions": {p: importlib.metadata.version(p) for p in (
            "nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi", "torch-c-dlpack-ext", "triton")},
        "timer": {"kind": "CUDA events around CUDA Graph replay", "calls_per_graph": CALLS_PER_GRAPH,
                  "warmup_ms": WARMUP_MS, "measurement_ms": MEASURE_MS, "rounds": ROUNDS,
                  "cache_regime": "steady-state same-buffer; not cold cache"},
        "timed_region": "prepared IDs / exact FA4 metadata and preferred layout -> preallocated BF16 O",
        "excluded": ["indexer", "metadata construction", "layout conversion", "allocations",
                     "input generation", "compilation", "correctness checks"],
        "our_cuda_modified": False, "fa4_source_modified": False,
        "sanitizer_run": False, "model_quality_claim": False,
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ours = stage_functions("p03")[1]
    if mode == "gate":
        configs = [
            ((1, 16, 1, 384, 2048), "all", 17, 1.0, False),
            ((1, 16, 1, 384, 4096), "disjoint", 29, 1.0, True),
            ((1, 64, 4, 384, 4096), "independent", 41, 4.0, True),
            ((2, 16, 1, 4096, 4096), "independent", 17, 1.0, False),
            ((2, 16, 1, 4096, 16384), "paired", 29, 0.25, False),
        ]
        for shape, pattern, seed, scale, reverse in configs:
            if time.monotonic() - started > 350:
                raise RuntimeError("Split remaining correctness cases; do not raise timeout=8*60")
            CURRENT["checks"].append(one_case(ours, shape, pattern, seed, timing=False,
                                              scale=scale, reverse=reverse))
    else:
        hashes = {}
        for pattern, seed in performance_cases(skv):
            if time.monotonic() - started > 350:
                raise RuntimeError("Split remaining benchmark cases; do not raise timeout=8*60")
            record = one_case(ours, (2, 16, 1, 4096, skv), pattern, seed, timing=True)
            if seed in hashes and record["input_sha256"] != hashes[seed]:
                raise AssertionError("paired and independent masks must use bit-identical Q/K/V")
            hashes[seed] = record["input_sha256"]
            CURRENT["records"].append(record)
    CURRENT.update(passed=True, elapsed_seconds=time.monotonic()-started, gpu_end=base.gpu_state())
    return CURRENT
