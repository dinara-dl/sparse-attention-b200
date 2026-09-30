"""Correctness, quality, stages and directly timed connected attention paths."""
import hashlib
import importlib.metadata
import statistics
import time

import torch
import benchmark as bench
from metadata import build_plan, make_lists
from block_selectors import SummarySelector, PreparedFP4Selector
from backends import DynamicFa4
import reference as ref

CURRENT = {"passed": False, "records": [], "checks": []}
CONFIGS = ((1, False), (1, True), (2, False), (2, True))


def sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def check_selector(selector):
    selector.run()
    qd, kd, scores, ids = ref.summary_reference(selector.q, selector.k, selector.r)
    torch.testing.assert_close(selector.qd, qd, atol=0, rtol=0)
    torch.testing.assert_close(selector.kd, kd, atol=0, rtol=0)
    torch.testing.assert_close(selector.scores, scores, atol=2e-5, rtol=3e-4)
    torch.testing.assert_close(selector.ids, ids, atol=0, rtol=0)
    return {"score_max_abs": (selector.scores - scores).abs().max().item(),
            "exact_ids": True, "exact_bf16_descriptors": True}


def check_fp4(selector):
    from sparse.fp4_indexer.fp4_reference import fp4_scores_reference
    from sparse.fp4_indexer.reference import tile_topk_reference
    selector.run()
    scores = fp4_scores_reference(selector.inputs)
    torch.testing.assert_close(selector.raw, scores, rtol=1e-4, atol=1e-3)
    ids = tile_topk_reference(scores, batch_size=selector.inputs["batch_size"], len_q=selector.inputs["len_q"])
    torch.testing.assert_close(selector.ids, ids, rtol=0, atol=0)
    return {"score_max_abs": (selector.raw - scores).abs().max().item(), "exact_ids": True,
            "environment_note": "unchanged scorer checked in the same DSL environment as FA4"}


def make_backends(q, k, v, ids):
    return {f"fa4_q{s}_{'pack' if p else 'unpack'}": DynamicFa4(q, k, v, ids, stage=s, pack=p)
            for s, p in CONFIGS}


def check_metadata(backend, lists):
    backend.prepare()
    expected = build_plan(lists, backend.nt, backend.stage)
    for actual_counts, actual_ids, reference_rows in ((backend.fc, backend.fi, expected.full),
                                                     (backend.pc, backend.pi, expected.partial)):
        cnt, ids = actual_counts.cpu().tolist(), actual_ids.cpu().tolist()
        for b in range(backend.b):
            for m, row in enumerate(reference_rows[b][0]):
                assert cnt[b][0][m] == len(row)
                assert sorted(ids[b][0][m][:len(row)]) == row
    if backend.stage == 2:
        assert backend.lookup.cpu().tolist() == expected.allowed


def timed(functions, warmup_ms=100, measure_ms=500, rounds=3):
    graphs = {name: bench.capture(fn) for name, fn in functions.items()}
    samples = {name: [] for name in functions}
    initial, orders = list(functions), []
    for rnd in range(rounds):
        order = initial[rnd:] + initial[:rnd]
        if rnd % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            ms = bench.time_graph(graphs[name], warmup_ms, measure_ms)
            samples[name].append(ms)
            print(f"TIME round={rnd} {name} ms={ms:.6f}", flush=True)
    return {"samples_ms": samples, "median_ms": {n: statistics.median(v) for n, v in samples.items()},
            "orders": orders, "warmup_ms": warmup_ms, "measure_ms": measure_ms, "rounds": rounds}


def gate(here, own_op):
    for family in ("gaussian", "cancellation_stress"):
        q, k, v = ref.make_inputs(1, 512, 4096, family, 17)
        for r in (1, 4):
            selector = SummarySelector(q, k, r)
            check = {"family": family, "r": r, "selector": check_selector(selector)}
            output = torch.empty_like(q)
            def ours_pipeline():
                selector.run()
                own_op(q, k, v, selector.ids, output)
            backends = make_backends(q, k, v, selector.ids)
            functions = {"ours": ours_pipeline}
            for name, backend in backends.items():
                def pipeline(backend=backend):
                    selector.run()
                    backend.prepare()
                    backend.run()
                functions[name] = pipeline
            original = q.clone()
            for name, pipeline in functions.items():
                pipeline()
                expected = bench.reference(q, k, v, selector.ids, sample=False)
                actual = output if name == "ours" else backends[name].logical_output()
                check[name] = bench.check_reference(actual, expected)
                graph, _ = bench.capture(pipeline, calls=1)
                q.copy_(original.roll(128, dims=2))
                expected_ids = ref.summary_reference(q, k, r)[3]
                graph.replay()
                torch.testing.assert_close(selector.ids, expected_ids, rtol=0, atol=0)
                changed_ref = bench.reference(q, k, v, expected_ids, sample=False)
                check[name + "_changed_graph"] = bench.check_reference(actual, changed_ref)
                q.copy_(original)
            CURRENT["checks"].append(check)
            print(f"GATE_OK {family} R{r} including changed-Q graph replay", flush=True)
    q, k, v = ref.make_inputs(1, 512, 4096, "gaussian", 29)
    for r in (1, 4):
        selector = SummarySelector(torch.zeros_like(q), torch.zeros_like(k), r)
        CURRENT["checks"].append({"zero_ties_r": r, **check_selector(selector)})
    ids = torch.tensor(make_lists(1, 1, 4, 32, "paired", 17), device="cuda", dtype=torch.int32)
    for stage, pack in CONFIGS:
        backend = DynamicFa4(q, k, v, ids, stage=stage, pack=pack)
        def call():
            backend.prepare()
            backend.run()
        graph, _ = bench.capture(call, calls=1)
        for pattern in ("disjoint", "paired", "independent"):
            lists = make_lists(1, 1, 4, 32, pattern, 29)
            ids.copy_(torch.tensor(lists, device="cuda", dtype=torch.int32))
            graph.replay()
            # Check graph-generated metadata BEFORE another eager prepare.
            expected = bench.reference(q, k, v, ids, sample=False)
            result = bench.check_reference(backend.logical_output(), expected)
            check_metadata(backend, lists)
            CURRENT["checks"].append({"metadata_transition": pattern, "stage": stage, "pack": pack, **result})
    fp4 = PreparedFP4Selector(here, 1, 256, 4096, 313)
    CURRENT["checks"].append({"fp4": check_fp4(fp4)})
    graph, _ = bench.capture(fp4.run, calls=1)
    fp4.inputs["q"].bitwise_xor_(0x77)
    graph.replay()
    captured_ids = fp4.ids.clone()
    check_fp4(fp4)
    torch.testing.assert_close(captured_ids, fp4.ids, rtol=0, atol=0)
    CURRENT["checks"].append({"fp4_changed_graph_exact_ids": True})


def evaluate(own_op, qkv, selector, dense=None):
    q, k, v = qkv
    selector_check = check_selector(selector) if isinstance(selector, SummarySelector) else check_fp4(selector)
    ids_initial = selector.ids.clone()
    lists = ids_initial.cpu().tolist()
    out = torch.empty_like(q)
    def ours_attention():
        own_op(q, k, v, selector.ids, out)
    def ours_pipeline():
        selector.run()
        ours_attention()
    ours_attention()
    expected = bench.reference(q, k, v, selector.ids, sample=False)
    checks = {"selector": selector_check, "ours": bench.check_reference(out, expected)}
    backends = make_backends(q, k, v, selector.ids)
    funcs = {"score": selector.score, "reduce_topk": selector.topk, "selector": selector.run,
             "ours_attention": ours_attention, "ours_pipeline": ours_pipeline}
    if isinstance(selector, SummarySelector):
        funcs["pooling"] = selector.prepare
    for name, backend in backends.items():
        check_metadata(backend, lists)
        backend.run(observe=True)
        checks[name] = bench.check_reference(backend.logical_output(), expected)
        funcs[name + "_attention"] = backend.run
        def pipeline(backend=backend):
            selector.run()
            backend.prepare()
            backend.run()
        funcs[name + "_pipeline"] = pipeline
    funcs["q2_metadata"] = backends["fa4_q2_unpack"].prepare
    result = {"selector": selector.name, "correctness": checks,
              "mask_stats": build_plan(lists, k.shape[2] // 128, 2).stats(),
              "selected_sha256": sha(ids_initial), "qkv_sha256": [sha(t) for t in qkv],
              "configs": {n: b.description() for n, b in backends.items()},
              "quality": ref.quality(ids_initial, expected, dense) if dense is not None else None,
              "input_boundary": "main BF16 Q/K" if dense is not None else "prepared independent FP4 proxies; quality N/A"}
    result.update(timing=timed(funcs), gpu_state=bench.gpu_state())
    torch.testing.assert_close(selector.ids, ids_initial, atol=0, rtol=0)
    result["post_graph_correctness"] = {"ours": bench.check_reference(out, expected)}
    for name, backend in backends.items():
        result["post_graph_correctness"][name] = bench.check_reference(backend.logical_output(), expected)
    medians = result["timing"]["median_ms"]
    best = min(backends, key=lambda n: medians[n + "_pipeline"])
    result["best_fa4_pipeline"] = best
    result["paired_pipeline_speedup"] = medians[best + "_pipeline"] / medians["ours_pipeline"]
    return result


def run(here, fa4_here, mode, skv):
    global CURRENT
    started = time.monotonic()
    CURRENT = {"mode": mode, "skv": skv, "records": [], "checks": [], "passed": False}
    bench.CURRENT = CURRENT
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    CURRENT.update(device=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                   cuda_version=torch.version.cuda, gpu_start=bench.gpu_state(),
                   versions={p: importlib.metadata.version(p) for p in (
                       "nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi", "triton")})
    own_op = bench.load_ours(fa4_here)
    if mode == "gate":
        gate(here, own_op)
    elif mode == "performance":
        for family in ("gaussian", "cancellation_stress"):
            qkv = ref.make_inputs(2, 4096, skv, family, 17)
            print(f"ORACLE_START {family} Skv={skv}", flush=True)
            dense = ref.oracle(*qkv)
            q, k, v = qkv
            dense_backends = {f"dense_{'pack' if pack else 'unpack'}_{'one' if one else 'auto'}":
                              DynamicFa4(q, k, v, dense=True, stage=2, pack=pack, one_cta=one)
                              for pack, one in ((False, True), (False, False), (True, True), (True, False))}
            for backend in dense_backends.values():
                backend.run(observe=True)
                bench.error_stats(backend.logical_output(), dense["output"])
            dense_time = timed({n: b.run for n, b in dense_backends.items()})
            best_dense = min(dense_time["median_ms"], key=dense_time["median_ms"].get)
            CURRENT.setdefault("dense", []).append({"family": family, "timing": dense_time, "best": best_dense,
                                                     "configs": {n: b.description() for n, b in dense_backends.items()}})
            for r in (1, 4):
                if time.monotonic() - started > 350:
                    CURRENT.update(incomplete=True, reason="split rather than exceed eight minutes")
                    return CURRENT
                print(f"SELECTOR_START {family} R{r} Skv={skv}", flush=True)
                result = evaluate(own_op, qkv, SummarySelector(q, k, r), dense)
                result.update(family=family, best_dense_ms=dense_time["median_ms"][best_dense])
                CURRENT["records"].append(result)
            if family == "gaussian":
                print(f"SELECTOR_START prepared FP4 Skv={skv}", flush=True)
                fp4 = PreparedFP4Selector(here, 2, 4096, skv, 313)
                result = evaluate(own_op, qkv, fp4)
                result.update(family=family, best_dense_ms=dense_time["median_ms"][best_dense])
                CURRENT["records"].append(result)
    else:
        raise ValueError(mode)
    CURRENT.update(passed=True, elapsed_seconds=time.monotonic() - started, gpu_end=bench.gpu_state(),
                   no_indexer_training=True, model_quality_claim=False,
                   timer="CUDA events, 8 calls/graph, warmup100ms, measure500ms, 3 rotated rounds; same-buffer",
                   full_pipeline="directly timed, includes all new BF16 pooling/scoring/TopK and necessary GPU metadata",
                   fp4_boundary="prepared proxies and reordered scales; excludes projection/quantization/preparation",
                   unchanged_attention_sources=True)
    return CURRENT
