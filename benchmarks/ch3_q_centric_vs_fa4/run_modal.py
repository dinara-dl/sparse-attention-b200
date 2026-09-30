"""Current p03 vs pinned FA4: one gate, then one eight-minute job per KV length."""

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/current_fa4")
ROOT = HERE.parents[2] if modal.is_local() else Path("/current_main")
BASE = HERE.parent / "fa4_same_mask" if modal.is_local() else Path("/fa4_same_mask")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "sparse"))
import modal_runner as historical
from progression import SOURCE_FILES as CORE_FILES

OWN_FILES = ("protocol.py", "current_benchmark.py", "run_modal.py", "test_protocol.py")
HELPER_FILES = ("metadata.py", "test_metadata.py", "benchmark.py", "fa4_adapter.py", "controls.py", "modal_runner.py")
image = historical.image
for name in CORE_FILES:
    image = image.add_local_file(ROOT / name, "/current_main/" + name)
for name in OWN_FILES:
    image = image.add_local_file(HERE / name, "/current_fa4/" + name)
app = modal.App("xomtention-current-p03-vs-fa4", image=image)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {**{"main/" + name: ROOT / name for name in CORE_FILES},
            **{"helpers/" + name: BASE / name for name in HELPER_FILES},
            **{"experiment/" + name: HERE / name for name in OWN_FILES}}


def manifest():
    return {name: sha(path) for name, path in sources().items()}


@app.function(gpu="B200", timeout=8 * 60, volumes={"/fa4_cache": historical.CACHE}, max_containers=1)
def experiment(mode: str, skv: int, provenance: dict):
    import time
    import traceback
    sys.path.insert(0, str(HERE))
    started = time.monotonic()
    result = {"passed": False, "mode": mode, "skv": skv}
    try:
        if manifest() != provenance["source_sha256"]:
            raise RuntimeError("mounted sources differ from the submitted manifest")
        import flash_attn.cute
        installed = Path(flash_attn.cute.__file__).resolve().parent
        actual = {str(p.relative_to(installed)): sha(p) for p in sorted(installed.rglob("*.py"))}
        if actual != provenance["fa4_python_sha256"]:
            raise RuntimeError("installed FA4 differs from the pinned clean checkout")
        import current_benchmark
        result = current_benchmark.run(mode, skv)
    except Exception:
        if "current_benchmark" in sys.modules:
            result = dict(sys.modules["current_benchmark"].CURRENT)
        result.update(passed=False, error=traceback.format_exc())
        print(result["error"], flush=True)
    result.update(provenance=provenance, modal_app_id=app.app_id,
                  remote_elapsed_seconds=time.monotonic()-started)
    historical.CACHE.commit()
    print(f"DONE mode={mode} skv={skv} passed={result['passed']}", flush=True)
    return result


@app.local_entrypoint()
def main(mode: str = "gate", skv: int = 4096):
    if mode not in ("gate", "performance") or skv not in (2048, 4096, 8192, 16384):
        raise ValueError("gate/performance; Skv=2048/4096/8192/16384")
    for path in sources().values():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    reference = historical.REFERENCE
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip()
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "flash_attn/cute"], cwd=reference).returncode == 0
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "flash_attn/cute"], cwd=reference, text=True).strip()
    if commit != historical.FA4_COMMIT or not clean or untracked:
        raise RuntimeError("FA4 must match the pinned clean source tree")
    cute = reference / "flash_attn/cute"
    provenance = {
        "source_sha256": manifest(), "fa4_commit": commit, "fa4_tracked_tree_clean": True,
        "fa4_python_sha256": {str(p.relative_to(cute)): sha(p) for p in sorted(cute.rglob("*.py"))},
        "our_version": "p03", "our_source": "sparse/02_gqa_block_sparse.cu",
        "our_cuda_sha256": sha(ROOT / "sparse/02_gqa_block_sparse.cu"),
        "timeout_seconds": 8 * 60,
    }
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    if mode == "performance":
        gates = []
        for path in sorted(results.glob("*_gate_*.json")):
            if path.stem.endswith("_call"):
                continue
            prior = json.loads(path.read_text())
            evidence = prior.get("provenance", {})
            if (prior.get("passed") and evidence.get("source_sha256") == provenance["source_sha256"]
                    and evidence.get("fa4_python_sha256") == provenance["fa4_python_sha256"]
                    and evidence.get("fa4_commit") == commit):
                gates.append(path.name)
        if not gates:
            raise RuntimeError("a source-matching successful correctness gate is required before timing")
        provenance["gate_file"] = gates[-1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{stamp}_{mode}_{skv}"
    for name, source in sources().items():
        target = results / "sources" / stem / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    target, ledger = results / (stem + ".json"), results / (stem + "_call.json")
    entry = {"status": "submitting", "mode": mode, "skv": skv, "provenance": provenance}
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    call = experiment.spawn(mode, skv, provenance)
    entry.update(status="running", call_id=call.object_id)
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"MODAL_CALL {call.object_id} ledger={ledger}", flush=True)
    try:
        result = call.get()
    except BaseException as error:
        entry.update(status="client_interrupted_remote_state_unknown", error=repr(error))
        ledger.write_text(json.dumps(entry, indent=2) + "\n")
        raise
    target.write_text(json.dumps(result, indent=2) + "\n")
    entry.update(status="finished", passed=result["passed"], result_file=target.name)
    ledger.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"SAVED {target} passed={result['passed']}", flush=True)
    if not result["passed"]:
        raise RuntimeError("GPU job failed; inspect saved evidence before continuing")
