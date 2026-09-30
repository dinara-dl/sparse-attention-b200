"""Explicit, serialized eight-minute B200 jobs, each saved with its call ID."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import modal

FA4_COMMIT = "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820"
HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/fa4_same_mask")
REMOTE = Path("/fa4_same_mask")
REFERENCE = HERE / "reference/flash-attention"
FA4_SOURCE = REFERENCE / "flash_attn/cute" if modal.is_local() else Path("/fa4_source")
OWN_FILES = ("metadata.py", "test_metadata.py", "fa4_adapter.py", "benchmark.py", "controls.py", "modal_runner.py")
BASE_FILES = ("10_native_gqa_block_sparse.cu", "native_gqa_attention.cpp", "common.cuh", "approximations.cuh", "smem_layout_swizzle128.cuh")
CACHE = modal.Volume.from_name("xomtention-fa4-same-mask-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("ninja", "numpy", "einops", "pybind11", "setuptools>=75", "setuptools-scm>=8", "typing_extensions",
                    "nvidia-cutlass-dsl[cu13]==4.6.2", "quack-kernels==0.6.4", "apache-tvm-ffi==0.1.12", "torch-c-dlpack-ext")
    .add_local_dir(FA4_SOURCE, "/fa4_source", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .run_commands("python -m pip install --no-deps /fa4_source")
    .env({"PYTHONDONTWRITEBYTECODE": "1", "MAX_JOBS": "4",
          "TORCH_EXTENSIONS_DIR": "/fa4_cache/torch",
          "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
          "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "/fa4_cache/cute"})
)
for name in OWN_FILES:
    image = image.add_local_file(HERE / name, str(REMOTE / name))
for name in BASE_FILES:
    image = image.add_local_file(HERE / "baseline" / name, str(REMOTE / "baseline" / name))
app = modal.App("xomtention-fa4-same-mask-stage1", image=image)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(here):
    return {name: sha(here / name) for name in (*OWN_FILES, *("baseline/" + x for x in BASE_FILES))}


@app.function(gpu="B200", timeout=8 * 60, volumes={"/fa4_cache": CACHE}, max_containers=1)
def experiment(mode: str, provenance: dict, warmup_ms: int, measure_ms: int, rounds: int):
    import sys
    import traceback
    import time
    sys.path.insert(0, str(REMOTE))
    started = time.monotonic()
    result = {"passed": False, "mode": mode}
    try:
        if source_manifest(REMOTE) != provenance["source_sha256"]:
            raise RuntimeError("uploaded first-party source differs from the local manifest")
        import flash_attn.cute
        installed = Path(flash_attn.cute.__file__).resolve().parent
        actual_tree = {str(p.relative_to(installed)): sha(p) for p in sorted(installed.rglob("*.py"))}
        if actual_tree != provenance["fa4_python_sha256"]:
            raise RuntimeError("installed FA4 Python sources differ from the pinned clean checkout")
        import benchmark
        if mode == "fairness_controls":
            import controls
            result = controls.run(REMOTE, warmup_ms, measure_ms, rounds)
        else:
            result = benchmark.run(REMOTE, mode, warmup_ms, measure_ms, rounds)
    except Exception:
        if "benchmark" in sys.modules:
            result = dict(sys.modules["benchmark"].CURRENT)
        result.update(passed=False, error=traceback.format_exc())
        print(result["error"], flush=True)
    result.update(provenance=provenance, modal_app_id=app.app_id,
                  remote_elapsed_seconds=time.monotonic() - started)
    CACHE.commit()
    print("RUN_RESULT " + json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(mode: str = "correctness", warmup_ms: int = 250, measure_ms: int = 1000, rounds: int = 3):
    import ast
    if mode not in ("correctness", "core_short", "core_long", "gqa_groups", "hot_confirmation", "fairness_controls"):
        raise ValueError("unknown stage-1 mode")
    for name in OWN_FILES:
        ast.parse((HERE / name).read_text(), filename=name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REFERENCE, text=True).strip()
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "flash_attn/cute"], cwd=REFERENCE).returncode == 0
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "flash_attn/cute"], cwd=REFERENCE, text=True).strip()
    clean = clean and not untracked
    if commit != FA4_COMMIT or not clean:
        raise RuntimeError("FA4 checkout must be clean and pinned")
    original = HERE.parents[1]
    for name in BASE_FILES:
        original_path = original / name if name.endswith((".cu", ".cpp")) else original.parent / name
        if sha(HERE / "baseline" / name) != sha(original_path):
            raise RuntimeError("main checkpoint changed since snapshot; review and deliberately rebaseline")
    provenance = {
        "fa4_commit": commit, "fa4_tracked_tree_clean": clean,
        "source_sha256": source_manifest(HERE),
        "fa4_python_sha256": {str(p.relative_to(REFERENCE / "flash_attn/cute")): sha(p)
                              for p in sorted((REFERENCE / "flash_attn/cute").rglob("*.py"))},
        "baseline": "main s10 native GQA (historical v15), byte-identical source snapshot",
        "first_party_timeout_seconds": 8 * 60,
    }
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    if mode != "correctness":
        valid_gate = False
        for candidate in results.glob("*_correctness.json"):
            prior = json.loads(candidate.read_text())
            evidence = prior.get("provenance", {})
            required = ("metadata.py", "fa4_adapter.py", "benchmark.py", *("baseline/" + x for x in BASE_FILES))
            if (prior.get("passed") and evidence.get("fa4_commit") == FA4_COMMIT
                    and all(evidence.get("source_sha256", {}).get(n) == provenance["source_sha256"][n] for n in required)):
                valid_gate = True
                provenance["correctness_gate_file"] = candidate.name
                break
        if not valid_gate:
            raise RuntimeError("a successful matching correctness-only GPU gate is required before timing")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = results / f"{stamp}_{mode}.json"
    ledger = results / f"{stamp}_{mode}_call.json"
    entry = {"mode": mode, "status": "submitting", "provenance": provenance}
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    call = experiment.spawn(mode, provenance, warmup_ms, measure_ms, rounds)
    entry.update(call_id=call.object_id, status="running")
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"MODAL_CALL {call.object_id} ledger={ledger}", flush=True)
    try:
        result = call.get()
    except BaseException as error:
        entry.update(status="client_interrupted_remote_state_unknown", error=repr(error))
        ledger.write_text(json.dumps(entry, indent=2) + "\n")
        raise
    target.write_text(json.dumps(result, indent=2) + "\n")
    entry.update(status="finished", passed=result["passed"], result_path=str(target))
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"SAVED {target} passed={result['passed']}", flush=True)
    if not result["passed"]:
        raise RuntimeError("stage did not pass; see saved result before launching any benchmark")
