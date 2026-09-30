"""Unmodified selectors, current p03, and directly measured online FA4 paths."""
import importlib.metadata
from pathlib import Path
import statistics
import time

import torch
import benchmark as bench
import harness as checks
import reference as ref
from block_selectors import SummarySelector, PreparedFP4Selector
from metadata import build_plan
from progression import stage_functions
from protocol import SEEDS, METHODS, WARMUP_MS, MEASURE_MS, ROUNDS, GRAPH_CALLS, rotated, order

HERE = Path(__file__).resolve().parent
CURRENT = {"passed": False, "records": [], "checks": []}


def fp4_selector(batch, sq, skv, seed):
    # The historical helper's optional frozen directory is deliberately absent.
    # Import resolution must use the current mounted main source, not its archive.
    assert not (HERE / "frozen").exists()
    result = PreparedFP4Selector(HERE, batch, sq, skv, seed)
    import sparse.fp4_indexer.fp4 as source
    assert Path(source.__file__).resolve() == Path("/current_main/sparse/fp4_indexer/fp4.py")
    return result


def paths(own_op, qkv, selector):
    q, k, v = qkv
    out = torch.empty_like(q)
    def attention():
        own_op(q, k, v, selector.ids, out)
    def ours():
        selector.run()
        attention()
    backends = checks.make_backends(q, k, v, selector.ids)
    functions = {"selector": selector.run, "ours_attention": attention, "ours_pipeline": ours}
    for name, backend in backends.items():
        def pipeline(backend=backend):
            selector.run()
            backend.prepare()
            backend.run()
        functions[name + "_pipeline"] = pipeline
    return out, backends, functions


def extra_fp4_gate(own_op):
    qkv = ref.make_inputs(1, 256, 4096, "gaussian", 41)
    selector = fp4_selector(1, 256, 4096, 315)
    out, backends, functions = paths(own_op, qkv, selector)
    for name, fn in functions.items():
        if not name.endswith("_pipeline"):
            continue
        graph, _ = bench.capture(fn, calls=1)
        selector.inputs["q"].bitwise_xor_(0x77)
        checks.check_fp4(selector)
        expected_ids = selector.ids.clone()
        expected = bench.reference(*qkv, expected_ids, sample=False)
        actual = out if name == "ours_pipeline" else backends[name.removesuffix("_pipeline")].logical_output()
        actual.fill_(float("nan"))
        selector.ids.fill_(-1)
        graph.replay()
        torch.testing.assert_close(selector.ids, expected_ids, atol=0, rtol=0)
        CURRENT["checks"].append({"fp4_changed_pipeline": name,
                                  **bench.check_reference(actual, expected)})


def evaluate(own_op, qkv, selector, seed_index, method_index):
    selector_check = checks.check_selector(selector) if isinstance(selector, SummarySelector) else checks.check_fp4(selector)
    initial_ids = selector.ids.clone()
    lists = initial_ids.cpu().tolist()
    expected = bench.reference(*qkv, initial_ids, sample=False)
    out, backends, functions = paths(own_op, qkv, selector)
    correctness = {"selector": selector_check}
    functions["ours_attention"]()
    correctness["ours"] = bench.check_reference(out, expected)
    for name, backend in backends.items():
        checks.check_metadata(backend, lists)
        backend.run(observe=True)
        correctness[name] = bench.check_reference(backend.logical_output(), expected)
    graphs = {name: bench.capture(fn, calls=GRAPH_CALLS) for name, fn in functions.items()}
    samples = {name: [] for name in functions}
    orders = []
    for rnd in range(ROUNDS):
        sequence = order(functions, seed_index, method_index, rnd)
        orders.append(sequence)
        for name in sequence:
            ms = bench.time_graph(graphs[name], WARMUP_MS, MEASURE_MS)
            samples[name].append(ms)
            print(f"TIME seed={SEEDS[seed_index]} {selector.name} round={rnd} {name} ms={ms:.6f}", flush=True)
    post = {}
    # Independently replay every graph with poisoned output/IDs. This checks the
    # captured computation, not merely whichever output the final timer left.
    for name, (graph, _) in graphs.items():
        if name == "ours_attention":
            selector.ids.copy_(initial_ids)
        else:
            selector.ids.fill_(-1)
        actual = None
        if name.startswith("ours_"):
            actual = out
        elif name.endswith("_pipeline"):
            actual = backends[name.removesuffix("_pipeline")].logical_output()
        if actual is not None:
            actual.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(selector.ids, initial_ids, atol=0, rtol=0)
        post[name] = bench.check_reference(actual, expected) if actual is not None else {"exact_ids": True}
    return {
        "selector": selector.name, "seed": SEEDS[seed_index],
        "proxy_seed": 313 + seed_index if not isinstance(selector, SummarySelector) else None,
        "family": "gaussian", "correctness": correctness, "post_graph_correctness": post,
        "selected_sha256": checks.sha(initial_ids), "qkv_sha256": [checks.sha(t) for t in qkv],
        "mask_stats": build_plan(lists, qkv[1].shape[2] // 128, 2).stats(),
        "configs": {name: b.description() for name, b in backends.items()},
        "timing": {"samples_ms": samples, "median_ms": {n: statistics.median(v) for n, v in samples.items()},
                   "orders": orders, "warmup_ms": WARMUP_MS, "measure_ms": MEASURE_MS,
                   "rounds": ROUNDS, "graph_calls": GRAPH_CALLS},
        "gpu_state": bench.gpu_state(),
    }


def run(mode, skv, seed=0):
    global CURRENT
    started = time.monotonic()
    CURRENT = {"passed": False, "mode": mode, "skv": skv, "seed_filter": seed, "records": [], "checks": []}
    checks.CURRENT = CURRENT
    bench.CURRENT = CURRENT
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    CURRENT.update(device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                   cuda_version=torch.version.cuda, gpu_start=bench.gpu_state(),
                   versions={p: importlib.metadata.version(p) for p in ("nvidia-cutlass-dsl", "quack-kernels", "triton")})
    own_op = stage_functions("p03")[1]
    if mode == "gate":
        checks.gate(HERE, own_op)
        extra_fp4_gate(own_op)
    else:
        for si, input_seed in enumerate(SEEDS):
            if seed and seed != input_seed:
                continue
            qkv = ref.make_inputs(2, 4096, skv, "gaussian", input_seed)
            for method in rotated(METHODS, si):
                if time.monotonic() - started > 385:
                    CURRENT.update(incomplete=True, reason="split by --seed; do not increase eight-minute timeout")
                    return CURRENT
                print(f"CASE_START skv={skv} seed={input_seed} method={method}", flush=True)
                selector = (fp4_selector(2, 4096, skv, 313 + si) if method == "fp4"
                            else SummarySelector(qkv[0], qkv[1], int(method[1:])))
                CURRENT["records"].append(evaluate(own_op, qkv, selector, si, METHODS.index(method)))
                del selector
    CURRENT.update(passed=True, elapsed_seconds=time.monotonic()-started, gpu_end=bench.gpu_state(),
                   our_version="p03", model_quality_claim=False,
                   workload="B2 Hq16 Hkv1 Sq4096 D128 BF16 noncausal TopK16 of K128; Gaussian synthetic",
                   timer="CUDA Graph, 8 calls, warmup250ms, measure1000ms, 3 rotated rounds",
                   input_layout="shared BHSD buffers; FA4 receives zero-copy strided BSHD views",
                   full_pipeline="directly timed selector -> GPU metadata if needed -> attention; no CPU ID read",
                   fp4_boundary="independent prepared synthetic FP4 proxies; projection/quantization/preparation excluded",
                   allocation_boundary="reused outputs and selector buffers; FP4 public scorer allocates raw scores during graph capture",
                   quality="not remeasured here; correctness is against selected-mask FP32 attention, not dense/model quality")
    return CURRENT
