"""Aggregate successful, source-matched current-p03 runs without cherry-picking seeds."""
import json
from pathlib import Path
import statistics
from protocol import SEEDS, LENGTHS

HERE = Path(__file__).resolve().parent
LABELS = {"mean1_bf16": "R1", "mean4_bf16": "R4", "fp4_prepared_synthetic": "FP4*"}


def aggregate(paths):
    cases, inputs, source_hashes = {}, [], None
    for path in sorted(paths):
        if path.stem.endswith("_call"):
            continue
        data = json.loads(path.read_text())
        if not data.get("passed") or data.get("mode") != "performance":
            continue
        hashes = data["provenance"]["source_sha256"]
        if source_hashes is not None and hashes != source_hashes:
            raise ValueError("mixed source versions: explicitly select one protocol before aggregation")
        source_hashes = hashes
        inputs.append(path.name)
        for record in data["records"]:
            key = (data["skv"], record["selector"], record["seed"])
            if key in cases:
                raise ValueError(f"duplicate case {key}; choose runs explicitly, do not silently take the fastest")
            cases[key] = record
    rows = []
    for skv in LENGTHS:
        for selector, label in LABELS.items():
            records = [cases[(skv, selector, seed)] for seed in SEEDS]
            names = records[0]["timing"]["median_ms"]
            values = {n: [r["timing"]["median_ms"][n] * 1000 for r in records] for n in names}
            medians = {n: statistics.median(v) for n, v in values.items()}
            best = min((n for n in names if n.startswith("fa4_")), key=medians.get)
            rows.append({"skv": skv, "selector": label, "median_us": medians,
                         "seed_medians_us": values, "best_fa4": best,
                         "pipeline_speedup": medians[best] / medians["ours_pipeline"],
                         "paired_seed_speedups": [a/b for a, b in zip(values[best], values["ours_pipeline"])],
                         "seed_spread_percent": {n: 100*(max(v)-min(v))/statistics.median(v) for n, v in values.items()},
                         "mask_stats_by_seed": [r.get("mask_stats") for r in records],
                         "max_round_spread_percent": max(
                             100 * (max(v)-min(v))/statistics.median(v)
                             for r in records for v in r["timing"]["samples_ms"].values())})
    return {"source_files": inputs, "source_sha256": source_hashes, "rows": rows,
            "case_count": len(cases), "aggregation": "median of round medians across 3 seeds per fixed configuration"}


def main():
    result = aggregate(HERE.joinpath("results").glob("*_performance_*.json"))
    HERE.joinpath("summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Fresh selector benchmark: current p03 vs FA4 sparse", "",
             "All times in microseconds; lower is better. Median across three input seeds.", "",
             "| KV | Selector | Selector only | p03 attention only | Selector → p03 | Selector → best FA4 | p03 speedup |", 
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        v = row["median_us"]
        lines.append(f"| {row['skv']//1024}K | {row['selector']} | {v['selector']:.2f} | "
                     f"{v['ours_attention']:.2f} | {v['ours_pipeline']:.2f} | "
                     f"{v[row['best_fa4']]:.2f} | {row['pipeline_speedup']:.3f}× |")
    lines += ["", "*FP4 starts from independent prepared synthetic proxies: projection, quantization and preparation excluded.",
              "Its boundary and information differ from R1/R4; this is not a selector-quality ranking.", "",
              "Full pipelines are measured directly, not sums of isolated timings. FA4 Q2 metadata is included.",
              "Common BHSD inputs with zero-copy BSHD views for FA4. See README for the full protocol.", "",
              "## Best tested FA4 configurations", ""]
    lines += [f"- {r['skv']//1024}K {r['selector']}: {r['best_fa4']}; largest within-case round spread {r['max_round_spread_percent']:.2f}%." for r in result["rows"]]
    lines += ["", "## Raw result files", ""] + ["- `results/" + n + "`" for n in result["source_files"]]
    HERE.joinpath("RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
