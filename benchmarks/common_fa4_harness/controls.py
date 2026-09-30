"""Fairness controls: vectorized exact partial mask and dense launch choices.

Separate from the original measurements: do not overwrite the scalar-adapter
results. No FA4 source or our CUDA kernel is edited.
"""
from contextlib import contextmanager
import json
import statistics
import time

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute import interface as fa

import benchmark as bench
from fa4_adapter import Fa4Backend


@cute.jit
def block_uniform_vector32(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    # All 32 adjacent columns are in the same aligned KV128 block. Selection
    # is constant inside that block, so one lookup supplies all 32 mask bits.
    lookup = aux_tensors[0]
    qb = cutlass.min(q_idx[0] // 128, lookup.shape[2] - 1)
    kb = cutlass.min(kv_idx[0] // 128, lookup.shape[3] - 1)
    enabled = lookup[batch[0], 0, qb, kb] != 0
    packed = cute.make_rmem_tensor(1, cutlass.Uint32)
    packed[0] = cutlass.Uint32(0xFFFFFFFF) if enabled else cutlass.Uint32(0)
    return packed.load()


block_uniform_vector32.__vec_size__ = 32


@contextmanager
def disable_two_cta(disable):
    original = fa.utils._get_disable_2cta_default
    if disable:
        fa.utils._get_disable_2cta_default = lambda is_fwd=False: True
    try:
        yield
    finally:
        fa.utils._get_disable_2cta_default = original


class ControlledBackend(Fa4Backend):
    def __init__(self, *args, vector=False, force_1cta=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector, self.force_1cta = vector, force_1cta
        if vector and self.mask is not None:
            self.mask = block_uniform_vector32

    def run(self, *, observe=False):
        with disable_two_cta(self.force_1cta):
            return super().run(observe=observe)

    def description(self):
        return {**super().description(), "block_uniform_vector32_mask": self.vector,
                "force_disable_2cta_launch_option": self.force_1cta}


def one_case(ours, shape, pattern, seed, *, timing, scale=1.0, warmup_ms=250, measure_ms=1000, rounds=3):
    q, k, v, ids, out, own, plans = bench.make_case(ours, shape, pattern, seed, scale, reverse=not timing)
    own()
    sparse_ref = bench.reference(q, k, v, ids, sample=timing)
    dense_ref = sparse_ref if shape[-1] == 2048 else bench.reference(q, k, v, ids, sample=timing, dense=True)
    record = {"shape": dict(zip(("batch", "hq", "hkv", "sq", "skv"), shape)),
              "pattern": pattern, "seed": seed, "qk_scale": scale, "topk": 16,
              "correctness": {"ours_vs_fp32": bench.check_reference(out, sparse_ref)}}
    backends = {}
    for stage, pack in ((2, True), (2, False), (1, True), (1, False)):
        name = f"fa4_q{stage}_{'vec_' if stage == 2 else ''}{'pack' if pack else 'unpack'}"
        backends[name] = ControlledBackend(q, k, v, plan=plans[stage], q_stage=stage, pack=pack, folded=True, vector=stage == 2)
    for pack, force in ((True, False), (False, False), (True, True), (False, True)):
        name = f"fa4_dense_{'pack' if pack else 'unpack'}_{'1cta' if force else 'autocta'}"
        backends[name] = ControlledBackend(q, k, v, pack=pack, folded=False, force_1cta=force)
    for name, backend in backends.items():
        backend.run(observe=True)
        record["correctness"][name] = bench.check_reference(backend.logical_output(), dense_ref if backend.is_dense else sparse_ref)
        if not backend.is_dense:
            bench.error_stats(backend.logical_output(), out)
        print(f"CONTROL_OK {shape} {pattern} {name}", flush=True)
    record["configs"] = {name: backend.description() for name, backend in backends.items()}
    if timing:
        funcs = {"ours": own, **{name: backend.run for name, backend in backends.items()}}
        graphs = {name: bench.capture(fn) for name, fn in funcs.items()}
        samples = {name: [] for name in funcs}
        order0 = list(funcs)
        record["gpu_before"] = bench.gpu_state()
        orders = []
        for rnd in range(rounds):
            order = order0[rnd:] + order0[:rnd]
            if rnd % 2:
                order = list(reversed(order))
            orders.append(order)
            for name in order:
                ms = bench.time_graph(graphs[name], warmup_ms, measure_ms)
                samples[name].append(ms)
                print(f"CONTROL_TIME {shape} {pattern} round={rnd} {name} ms={ms:.6f}", flush=True)
        # Verify graph replay as well as ordinary calls, using the same references.
        record["post_graph_correctness"] = {"ours": bench.check_reference(out, sparse_ref)}
        for name, backend in backends.items():
            record["post_graph_correctness"][name] = bench.check_reference(backend.logical_output(), dense_ref if backend.is_dense else sparse_ref)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        best_sparse = min((name for name in medians if name.startswith("fa4_q")), key=medians.get)
        best_dense = min((name for name in medians if name.startswith("fa4_dense")), key=medians.get)
        flops = 4 * shape[0] * shape[1] * shape[3] * 2048 * 128
        record.update(samples_ms=samples, median_ms=medians, measurement_orders=orders,
                      best_fa4_sparse=best_sparse, best_fa4_dense=best_dense,
                      speedup_vs_best_fa4_sparse=medians[best_sparse] / medians["ours"],
                      speedup_vs_best_fa4_dense=medians[best_dense] / medians["ours"],
                      our_useful_tflops=flops / (medians["ours"] * 1e9), gpu_after=bench.gpu_state())
    print("CONTROL_RESULT " + json.dumps(record), flush=True)
    return record


def run(here, warmup_ms, measure_ms, rounds):
    start = time.monotonic()
    current = {"mode": "fairness_controls", "records": [], "checks": [], "controls": []}
    bench.CURRENT = current
    torch.backends.cuda.matmul.allow_tf32 = False
    current.update(device=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                   cuda_version=torch.version.cuda, gpu_start=bench.gpu_state())
    ours = bench.load_ours(here)
    for shape, pattern, scale in (((1, 16, 1, 384, 4096), "disjoint", 1.0),
                                  ((1, 64, 4, 384, 4096), "independent", 4.0)):
        current["checks"].append(one_case(ours, shape, pattern, 41, timing=False, scale=scale))
    cases = [((2, 16, 1, 4096, 2048), "all")]
    cases += [((2, 16, 1, 4096, skv), "independent") for skv in (4096, 8192, 16384)]
    cases += [((2, 64, 4, 4096, skv), "independent") for skv in (4096, 16384)]
    cases += [((2, 16, 1, 4096, 16384), "hot")]
    for shape, pattern in cases:
        if time.monotonic() - start > 370:
            current.update(passed=False, incomplete=True, reason="split controls rather than exceed eight minutes")
            return current
        current["controls"].append(one_case(ours, shape, pattern, 17, timing=True,
                                           warmup_ms=warmup_ms, measure_ms=measure_ms, rounds=rounds))
    current.update(passed=True, elapsed_seconds=time.monotonic() - start,
                   gpu_end=bench.gpu_state(), warmup_ms=warmup_ms, measure_ms=measure_ms, rounds=rounds,
                   timer="same CUDA event / graph timer as primary measurements; post-replay correctness checked",
                   purpose="exclude slow scalar mask adapter and suboptimal dense launch settings as explanations",
                   reference_source_modified=False, our_cuda_modified=False)
    return current
