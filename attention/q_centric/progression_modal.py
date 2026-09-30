"""Bounded B200 correctness and same-input benchmarks for dense14 main p02/p03."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/xomtention_progression")
REMOTE_ROOT = Path("/xomtention_progression")
sys.path.insert(0, str(LOCAL_ROOT / "sparse"))
from progression import SOURCE_FILES as CORE_FILES
from checkpoints import MAIN_SOURCES

FILES = (*CORE_FILES, "sparse/progression_validation.py", "sparse/progression_modal.py")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja", "numpy")
)
for filename in FILES:
    image = image.add_local_file(LOCAL_ROOT / filename, str(REMOTE_ROOT / filename))
app = modal.App("xomtention-dense14-main-progression", image=image)


def manifest(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def gpu_state():
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,"
                        "power.draw,clocks.current.sm,clocks.current.memory,pstate",
                        "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}


def capture(fn):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(8):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def time_graph(graph):
    import math
    import torch
    def timed(count):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)
    estimate = max(timed(8) / 8, 0.001)
    timed(max(1, math.ceil(75 / estimate)))
    count = max(8, math.ceil(250 / estimate))
    return timed(count) / (count * 8)


def benchmark_cases(lengths, started):
    import statistics
    import time
    import torch
    from progression import stage_functions
    from progression_validation import make_inputs, make_ids, reference, check_output
    records = []
    for skv in lengths:
        for seed in (17, 29, 41):
            for hkv, versions in ((16, ("p02",)), (1, ("p02", "p03"))):
                if time.monotonic() - started > 365:
                    raise RuntimeError("Split the remaining lengths; keep timeout=8*60")
                shape = (2, 16, hkv, 4096, skv)
                q, k, v = make_inputs(shape, seed)
                ids = make_ids(2, hkv, 32, skv//128, 16, seed+1000)
                expected = reference(q, k, v, ids)
                outputs = {key: torch.full_like(q, float("nan")) for key in versions}
                functions = {}
                checks = {}
                for key in versions:
                    out_fn = stage_functions(key)[1]
                    functions[key] = lambda out_fn=out_fn, key=key: out_fn(q, k, v, ids, outputs[key])
                    functions[key]()
                    checks[key] = check_output(outputs[key], expected)
                for key in versions[1:]:
                    torch.testing.assert_close(outputs[key], outputs[versions[0]], rtol=0, atol=0)
                graphs = {key: capture(fn) for key, fn in functions.items()}
                samples, orders = {key: [] for key in versions}, []
                before = gpu_state()
                for rnd in range(5):
                    shift = rnd % len(versions)
                    order = versions[shift:] + versions[:shift]
                    orders.append(list(order))
                    for key in order:
                        ms = time_graph(graphs[key])
                        samples[key].append(ms)
                        print(f"TIME skv={skv} hkv={hkv} seed={seed} round={rnd} {key}={ms:.6f}ms", flush=True)
                for key in versions:
                    checks[key + "_after_graph"] = check_output(outputs[key], expected)
                records.append({
                    "shape": list(shape), "topk": 16, "seed": seed, "pattern": "independent",
                    "same_input_storage": True, "cross_backend_exact": True, "correctness": checks,
                    "selected_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                    "samples_ms": samples, "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                    "orders": orders, "gpu_before": before, "gpu_after": gpu_state(),
                    "useful_sparse_qk_pv_flops": 4*2*16*4096*(16*128)*128,
                })
    rows = []
    for skv in lengths:
        for hkv in (16, 1):
            cases = [r for r in records if r["shape"][2] == hkv and r["shape"][4] == skv]
            rows.append({"Skv": skv, "Hkv": hkv,
                         "median_us": {key: statistics.median(r["median_ms"][key] for r in cases)*1000
                                       for key in cases[0]["median_ms"]}})
    return {"records": records, "summary": rows}


@app.function(gpu="B200", timeout=8 * 60, max_containers=1)
def run_gpu(mode: str, versions: list[str], lengths: list[int], provenance: dict):
    import time
    import traceback
    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "sparse"))
    from progression import load_ops
    from progression_validation import check_stage_correctness
    started = time.monotonic()
    result = {"passed": False, "mode": mode, "versions": versions, "lengths": lengths,
              "provenance": provenance, "modal_app_id": app.app_id,
              "device": torch.cuda.get_device_name(0), "gpu_before": gpu_state(),
              "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
              "protocol": "dense14_main_graph8_75_250_5rounds_3seeds_v1",
              "indexer_timed": False, "output_preallocated": True,
              "warmup_ms": 75, "measurement_ms": 250, "rounds": 5,
              "aggregation": "median of five rounds per seed, then median of three seeds",
              "excluded": ["compilation", "inputs", "ID generation", "allocations", "indexer", "FA4", "MSA"]}
    try:
        if manifest(REMOTE_ROOT) != provenance["source_sha256"]:
            raise RuntimeError("Mounted sources differ from local provenance")
        torch.backends.cuda.matmul.allow_tf32 = False
        load_ops()
        if mode == "gate":
            result["checks"] = {}
            for version in versions:
                if time.monotonic() - started > 365:
                    raise RuntimeError("Split remaining gate versions; keep timeout=8*60")
                result["checks"][version] = check_stage_correctness(version)
        else:
            result.update(benchmark_cases(lengths, started))
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    result.update(elapsed_seconds=time.monotonic()-started, gpu_after=gpu_state())
    return result


@app.local_entrypoint()
def main(mode: str = "gate", versions: str = "p02,p03", lengths: str = "4096,16384"):
    if mode not in ("gate", "benchmark"):
        raise ValueError("mode must be gate or benchmark")
    keys = [v.strip() for v in versions.split(",")]
    sizes = [int(v) for v in lengths.split(",")]
    if not keys or len(keys) != len(set(keys)) or any(v not in MAIN_SOURCES for v in keys):
        raise ValueError("versions must be distinct p02,p03; archived p01 uses benchmark_block_sparse_ms.py")
    if not sizes or len(sizes) != len(set(sizes)) or any(v not in (2048,4096,8192,16384) for v in sizes):
        raise ValueError("lengths must be distinct 2048/4096/8192/16384")
    # A benchmark includes both main stages on matched GQA inputs, plus p02 MHA.
    if mode == "benchmark" and set(keys) != set(MAIN_SOURCES):
        raise ValueError("benchmark requires p02,p03; split by lengths, not versions")
    hashes = manifest(LOCAL_ROOT)
    directory = LOCAL_ROOT / "sparse/results/dense14_progression"
    directory.mkdir(parents=True, exist_ok=True)
    gates = []
    covered = set()
    if mode == "benchmark":
        for path in sorted(directory.glob("*_gate.json")):
            data = json.loads(path.read_text())
            if data.get("passed") and data["provenance"]["source_sha256"] == hashes:
                covered.update(data["checks"])
                gates.append(path.name)
        if covered != set(MAIN_SOURCES):
            raise RuntimeError("Successful source-matching gates for p02,p03 are required")
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + mode
    provenance = {"source_sha256": hashes, "gate_files": gates, "timeout_seconds": 8*60}
    snapshot = directory / "sources" / stem
    for name in FILES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((LOCAL_ROOT / name).read_bytes())
    ledger = directory / (stem + "_call.json")
    call = run_gpu.spawn(mode, keys, sizes, provenance)
    entry = {"call_id": call.object_id, "status": "submitted", "provenance": provenance}
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    try:
        result = call.get()
    except Exception as exc:
        entry.update(status="failed", error=str(exc))
        ledger.write_text(json.dumps(entry, indent=2) + "\n")
        raise
    target = directory / (stem + ".json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    entry["status"] = "succeeded" if result["passed"] else "failed"
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"RESULT {target}")
    print(json.dumps(result.get("summary", {"passed": result["passed"], "versions": keys}), indent=2))
    if not result["passed"]:
        raise RuntimeError(result.get("error", "GPU validation failed"))
