"""Aggregate successful current-source runs; never pool historical s10 data."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from protocol import LENGTHS, PROTOCOL, SEEDS, performance_cases


def summarize(here):
    all_runs, chosen = [], {}
    for path in sorted((here / "results").glob("*.json")):
        if path.name.endswith("_call.json"):
            continue
        run = json.loads(path.read_text())
        all_runs.append((path, run))
        if run.get("passed") and run.get("mode") == "performance":
            chosen[run["skv"]] = (path, run)
    if set(chosen) != set(LENGTHS):
        raise RuntimeError("all four KV lengths must finish before publishing the summary")
    root = here.parents[2]
    current_sha = hashlib.sha256((root / "sparse/02_gqa_block_sparse.cu").read_bytes()).hexdigest()
    source_manifests = []
    groups = defaultdict(list)
    selected_runs = []
    for skv in LENGTHS:
        path, run = chosen[skv]
        if run["protocol"] != PROTOCOL or run["provenance"]["our_cuda_sha256"] != current_sha:
            raise RuntimeError("result protocol or p03 CUDA does not match the current source")
        source_manifests.append(run["provenance"]["source_sha256"])
        cases = [(r["pattern"], r["seed"]) for r in run["records"]]
        if cases != performance_cases(skv):
            raise RuntimeError(f"incomplete or duplicated cases for Skv={skv}")
        for r in run["records"]:
            if set(r["post_graph_correctness"]) != set(r["median_ms"]):
                raise RuntimeError("not every measured backend passed graph-replay correctness")
            groups[skv, r["pattern"]].append(r)
        selected_runs.append((path, run))
    if any(m != source_manifests[0] for m in source_manifests):
        raise RuntimeError("cannot pool runs from different first-party sources")
    entries = []
    lines = [
        "# Current p03 vs pinned FA4: measured results", "",
        "B200; B2, Hq/Hkv=16/1, Sq4096, D128, BF16, non-causal, TopK16, KV128.",
        "Current `sparse/02_gqa_block_sparse.cu`, through the production p03 loader.",
        f"CUDA SHA-256: `{current_sha}`.",
        "FA4: `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`; source tree unchanged.", "",
        "Latency in microseconds: median of per-seed medians (three rounds each; seeds17/29/41).",
        "Best FA4 is the lowest aggregate latency among four tested sparse / four dense configurations.",
        "Ratios are medians of within-seed ratios for that selected configuration, not ratios across GPU allocations.", "",
        "| Skv | Mask | Selected % | Current p03 µs | Best FA4 sparse µs | FA4 sparse / p03 | Best FA4 dense µs | FA4 dense / p03 |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for (skv, pattern), records in sorted(groups.items()):
        names = list(records[0]["median_ms"])
        medians = {name: statistics.median(r["median_ms"][name] for r in records) for name in names}
        best_sparse = min((name for name in names if name.startswith("fa4_q")), key=medians.get)
        best_dense = min((name for name in names if name.startswith("fa4_dense")), key=medians.get)
        ratios = {kind: [r["median_ms"][name] / r["median_ms"]["ours"] for r in records]
                  for kind, name in (("sparse", best_sparse), ("dense", best_dense))}
        entry = {
            "skv": skv, "pattern": pattern, "seeds": [r["seed"] for r in records],
            "median_us": {name: value * 1000 for name, value in medians.items()},
            "best_fa4_sparse": best_sparse, "best_fa4_dense": best_dense,
            "speedup_ratios": {kind: {"median": statistics.median(values), "min": min(values), "max": max(values)}
                               for kind, values in ratios.items()},
            "seed_ranges_us": {name: [min(r["median_ms"][name] for r in records)*1000,
                                       max(r["median_ms"][name] for r in records)*1000] for name in names},
            "our_useful_tflops": statistics.median(r["our_useful_tflops"] for r in records),
        }
        entries.append(entry)
        lines.append(f"| {skv} | {pattern} | {2048/skv*100:.1f} | {medians['ours']*1000:.2f} | "
                     f"{medians[best_sparse]*1000:.2f} | {statistics.median(ratios['sparse']):.3f}× | "
                     f"{medians[best_dense]*1000:.2f} | {statistics.median(ratios['dense']):.3f}× |")
    lines += ["", "## All tested configurations", "",
              "`q1` is an explicit Python launch override; Q2 partial blocks use the vector32 exact mask.",
              "Dense `1cta` explicitly disables 2CTA; `autocta` retains the pinned FA4 choice.", "",
              "| Skv | Mask | p03 | Q1 unpack | Q1 pack | Q2 unpack | Q2 pack | Dense unpack 1CTA | Dense pack 1CTA | Dense unpack auto | Dense pack auto |",
              "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = ("ours", "fa4_q1_unpack", "fa4_q1_pack", "fa4_q2_unpack", "fa4_q2_pack",
             "fa4_dense_unpack_1cta", "fa4_dense_pack_1cta", "fa4_dense_unpack_autocta", "fa4_dense_pack_autocta")
    for e in entries:
        lines.append(f"| {e['skv']} | {e['pattern']} | " + " | ".join(f"{e['median_us'][n]:.2f}" for n in order) + " |")
    gate_names = {run["provenance"]["gate_file"] for _, run in selected_runs}
    gate_runs = [(p, r) for p, r in all_runs if p.name in gate_names]
    if len(gate_runs) != len(gate_names) or any(not r.get("passed") for _, r in gate_runs):
        raise RuntimeError("matching correctness gate missing")
    used = gate_runs + selected_runs
    checks = [r for _, run in used for r in run.get("checks", []) + run.get("records", [])]
    max_error = max(stats["max_abs"] for r in checks for field in ("correctness", "post_graph_correctness")
                    for name, stats in r[field].items() if field == "post_graph_correctness" or name.endswith("_vs_fp32"))
    records = [r for _, run in selected_runs for r in run["records"]]
    our_gate_error = max(c["correctness"]["ours_vs_fp32"]["max_abs"]
                         for _, run in gate_runs for c in run["checks"])
    our_timed_error = max(r[field][key]["max_abs"] for r in records
                          for field, key in (("correctness", "ours_vs_fp32"), ("post_graph_correctness", "ours")))
    spreads = {name: max((max(r["samples_ms"][name])-min(r["samples_ms"][name]))/
                         statistics.median(r["samples_ms"][name])*100 for r in records) for name in order}
    calls = sum(len(v) for r in records for v in r["samples_ms"].values())
    lines += ["", "## Validation and timing", "",
              f"- {sum(len(r.get('checks', [])) for _, r in gate_runs)} full correctness-gate cases; {len(records)} measured input/mask cases; {calls} backend round timings.",
              f"- Full FP32 references for all outputs before timing and after replay. Maximum absolute reference error across these runs: {max_error:.8f}.",
              f"- p03 alone: maximum absolute reference error {our_gate_error:.8f} in the gate (including scale4), {our_timed_error:.8f} on the timed scale1 inputs.",
              "- Tolerance is the combined criterion abs(error) <= 0.02 + 0.02*abs(reference), not a standalone 0.02 maximum.",
              f"- Largest three-round min/max spread relative to the median: p03 {spreads['ours']:.3f}%; any backend {max(spreads.values()):.3f}%.",
              "- Graph8, warmup250ms, measurement1000ms, three rounds and seeds17/29/41; outputs preallocated.",
              "- Same GPU for all backends/seeds/patterns at one length; different lengths may use different physical B200s.",
              "- Paired and independent patterns use bit-identical Q/K/V for each seed. Full/partial metadata preserves the exact mask.",
              "- Same-buffer steady state, not cold-cache serving. Inputs, indexer, metadata, layouts, allocations, compilation and reference checks excluded.",
              "- Dense has a different mask except the 2K all-block control. No model-quality, end-to-end selector, sanitizer or profiler claim.",
              "- All first-party GPU functions retain timeout=8*60; CUDA sources and historical results are untouched.", "",
              "## Evidence", ""]
    for path, run in used:
        lines.append(f"- [{path.name}](results/{path.name}): {run['remote_elapsed_seconds']:.1f}s remote; app `{run['modal_app_id']}`.")
    omitted = [(p, r) for p, r in all_runs if p.name not in {p.name for p, _ in used}]
    for path, run in omitted:
        lines.append(f"- [{path.name}](results/{path.name}): not pooled; passed={run.get('passed')}.")
    result = {"protocol": PROTOCOL, "our_cuda_sha256": current_sha, "summary": entries,
              "timed_cases": len(records), "round_timings": calls, "max_reference_abs_error": max_error,
              "max_our_gate_abs_error": our_gate_error, "max_our_timed_abs_error": our_timed_error,
              "max_round_spread_percent": spreads, "remote_seconds": sum(r["remote_elapsed_seconds"] for _, r in used),
              "run_files": [p.name for p, _ in used]}
    (here / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (here / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    summarize(Path(__file__).resolve().parent)
