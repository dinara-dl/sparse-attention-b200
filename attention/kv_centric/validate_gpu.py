"""Correctness only: no warmup sweep, timings, TFLOP/s or MSA benchmarking."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from api import ROOT, attention, load_extension
from metadata import build_plan, make_selected
from reference import attention_reference, combine_reference

OUTPUT_ATOL, OUTPUT_RTOL = 0.02, 0.02
LSE_ATOL, LSE_RTOL = 0.002, 0.0002


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.iterdir()) if p.suffix in (".cu", ".cuh", ".cpp", ".py")}


def checked(actual, expected, *, lse=False):
    if not torch.isfinite(actual).all().item():
        raise AssertionError("non-finite result / unwritten poisoned workspace")
    torch.testing.assert_close(actual.float(), expected.float(),
        atol=LSE_ATOL if lse else OUTPUT_ATOL,
        rtol=LSE_RTOL if lse else OUTPUT_RTOL)
    return float((actual.float() - expected.float()).abs().max().item())


def check_combine():
    torch.manual_seed(951)
    partial = torch.randn(2, 2, 5, 16, 128, device="cuda")
    lses = torch.randn(2, 2, 5, 16, device="cuda") * 20
    # Large positive and negative logits test stable weighting (not averaging).
    lses[0, 0, 0].fill_(-1000)
    lses[0, 0, 0, 5] = 1000
    out, lse = load_extension().combine(partial, lses)
    expected, expected_lse = combine_reference(partial, lses)
    return {"output_max_abs": checked(out, expected),
            "lse_max_abs": checked(lse, expected_lse, lse=True)}


def run_case(name, b, hkv, g, sq, skv, kind, seed, transform=None, check_reorder=False):
    torch.manual_seed(seed)
    q = torch.randn((b, hkv * g, sq, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((b, hkv, skv, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    ids = make_selected(b, hkv, sq, skv // 128, kind, seed)
    if transform == "zero_q":
        q.zero_()
    elif transform == "large_qk":
        q.mul_(4)
        k.mul_(4)
    elif transform == "constant_v":
        for bi in range(b):
            for hi in range(hkv):
                v[bi, hi].fill_(1 + bi * hkv + hi)
    elif transform == "disjoint_rows":
        q.zero_()
        k.zero_()
        v.zero_()
        v[:, :, 2048:].fill_(1)
        for qi in range(sq):
            ids[:, :, qi] = torch.arange(16, dtype=torch.int32) + (qi % 2) * 16
    elif transform == "dominant_block":
        q.fill_(1)
        k.zero_()
        k[:, :, :128].fill_(4)
        # With 'hot', every query selects block zero, but at varying ranks below.
        for block in range(skv // 128):
            v[:, :, block * 128:(block + 1) * 128].fill_(block)
        for qi in range(sq):
            ids[:, :, qi] = torch.roll(ids[:, :, qi], shifts=qi % 16, dims=-1)

    plan = build_plan(ids, skv // 128, g)
    expected, expected_lse = attention_reference(q, k, v, ids)
    output, lse, partial, plse = attention(q, k, v, ids, plan=plan, return_debug=True)
    torch.cuda.synchronize()
    if not torch.isfinite(partial).all().item() or not torch.isfinite(plse).all().item():
        raise AssertionError(f"{name}: partial slots not all finite")
    result = {"name": name, "shape": {"B": b, "Hq": hkv*g, "Hkv": hkv,
        "Sq": sq, "Skv": skv, "D": 128, "TopK": 16}, "pattern": kind,
        "tasks": plan.tasks.shape[0], "edges": plan.query_ids.numel(),
        "output_max_abs": checked(output, expected),
        "lse_max_abs": checked(lse, expected_lse, lse=True),
        "all_partial_slots_written": True}

    # Independent audit of the actual GPU Combine using its actual partials.
    from_partials, from_partials_lse = combine_reference(partial, plse)
    result["combine_max_abs"] = checked(output, from_partials)
    checked(lse, from_partials_lse, lse=True)

    # Check actual GPU main's local LSE on several ranks, heads and edge/tail rows.
    max_partial_lse_error = 0.0
    max_partial_o_error = 0.0
    for bi, head, qi, rank in ((0, 0, 0, 0), (b-1, hkv*g-1, sq-1, 15),
                               (0, min(1, hkv*g-1), sq//2, 7)):
        kh = head // g
        block = int(ids[bi, kh, qi, rank])
        kb = k[bi, kh, block*128:(block+1)*128].float()
        vb = v[bi, kh, block*128:(block+1)*128].float()
        scores = q[bi, head, qi].float() @ kb.T / (128 ** 0.5)
        expected_po = torch.softmax(scores, -1) @ vb
        expected_plse = torch.logsumexp(scores, -1)
        max_partial_lse_error = max(max_partial_lse_error,
            checked(plse[bi, head, qi, rank], expected_plse, lse=True))
        max_partial_o_error = max(max_partial_o_error,
            checked(partial[bi, head, qi, rank], expected_po))
    result["sample_partial_lse_max_abs"] = max_partial_lse_error
    result["sample_partial_o_max_abs"] = max_partial_o_error
    if check_reorder:
        reordered = ids.flip(-1).contiguous()
        out2 = attention(q, k, v, reordered)
        result["reordered_output_max_abs"] = checked(out2, expected)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out3 = attention(q, k, v, ids, plan=plan)
            done = torch.cuda.Event()
            done.record()
        torch.cuda.current_stream().wait_event(done)
        result["non_default_stream_max_abs"] = checked(out3, expected)
        out4 = attention(q, k, v, ids, plan=plan)
        # Repeated same arithmetic/metadata must be deterministic in this v0.
        if not torch.equal(out4, output):
            raise AssertionError("identical inputs changed output across repeated calls")
        result["repeat_bitwise_equal"] = True
    print("KV0_CASE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def run_suite(smoke=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("run the correctness suite on B200 / SM100")
    report = {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "python": sys.version,
              "source_hashes": source_hashes(), "timing": "not measured",
              "metadata": "CPU reverse CSR/fixed chunks, uploaded before main",
              "tolerances": {"output_atol": OUTPUT_ATOL, "output_rtol": OUTPUT_RTOL,
                             "lse_atol": LSE_ATOL, "lse_rtol": LSE_RTOL}}
    load_extension()
    report["combine_unit"] = check_combine()
    if smoke:
        specs = [("g16_tail_smoke", 1, 1, 16, 9, 4096, "independent", 31, None)]
    else:
        specs = [
            ("g16_initial", 1, 1, 16, 256, 4096, "independent", 31, None),
            ("g1_batch_heads_tail", 2, 2, 1, 129, 4096, "independent", 32, None),
            ("g2_overlap", 1, 2, 2, 17, 4096, "overlap", 33, None),
            ("g4_batch_hot", 2, 2, 4, 9, 4096, "hot", 34, None),
            ("g8_tile_shared", 1, 2, 8, 19, 4096, "tile_shared", 35, None),
            ("g16_multi_group", 2, 2, 16, 9, 4096, "independent", 36, None),
            ("g16_hot_chunks_empty_blocks", 1, 1, 16, 257, 4096, "identical", 37, None),
            ("g1_single_query", 1, 1, 1, 1, 4096, "independent", 38, None),
            ("g16_single_query", 1, 1, 16, 1, 4096, "hot", 39, None),
            ("g16_two_shared_tiles", 1, 1, 16, 256, 4096, "tile_shared", 40, None),
            ("g16_dense_equivalence", 1, 1, 16, 13, 2048, "all_blocks", 41, None),
            ("g16_zero_q", 1, 1, 16, 9, 4096, "independent", 42, "zero_q"),
            ("g16_constant_v", 2, 2, 16, 9, 4096, "independent", 43, "constant_v"),
            ("g16_large_qk", 1, 1, 16, 17, 4096, "independent", 44, "large_qk"),
            ("g16_dominant_block", 1, 1, 16, 19, 4096, "hot", 45, "dominant_block"),
            ("g16_disjoint_rows", 1, 1, 16, 17, 4096, "independent", 46, "disjoint_rows"),
        ]
    report["cases"] = [run_case(*spec, check_reorder=(index == 0))
                       for index, spec in enumerate(specs)]
    report["passed"] = True
    report["max_output_abs"] = max(c["output_max_abs"] for c in report["cases"])
    report["max_lse_abs"] = max(c["lse_max_abs"] for c in report["cases"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print("KV0_RESULT " + json.dumps(run_suite(smoke=args.smoke), sort_keys=True), flush=True)
