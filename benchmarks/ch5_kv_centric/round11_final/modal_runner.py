"""Conservative round11: candidates, then confirmation; one spare call if needed."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

REMOTE_BASE=Path("/kv_centric");REMOTE=REMOTE_BASE/"optimization_round11"
HERE=Path(__file__).resolve().parent if modal.is_local() else REMOTE
BASE=HERE.parent
BASE_FILES=("primitives.cuh","metadata.py","reference.py","api.py","validate_gpu.py",
            "optimization_round4/bench_api4.py","optimization_round4/gpu_plan.py",
            "optimization_round4/planning.py","optimization_round4/timing.py",
            "optimization_round4/gpu_metadata.cu",
            "optimization_round7/timing7.py",
            "optimization_round8/combine8.cu","optimization_round9/split9_kernel.cu",
            "optimization_round1/combine_warp.cu")
NEW_FILES=("api11.py","benchmark11.py","tma11_kernel.cu","combine11.cu","wrapper11.cpp","modal_runner.py","test_local.py")
image=(modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",add_python="3.12")
       .entrypoint([]).uv_pip_install("torch==2.9.1",index_url="https://download.pytorch.org/whl/cu130")
       .uv_pip_install("ninja","numpy").env({"PYTHONDONTWRITEBYTECODE":"1","MAX_JOBS":"2"}))
for n in BASE_FILES:image=image.add_local_file(BASE/n,str(REMOTE_BASE/n))
for n in NEW_FILES:image=image.add_local_file(HERE/n,str(REMOTE/n))
CACHE=modal.Volume.from_name("xomtention-kv-round11-build",create_if_missing=True)
app=modal.App("xomtention-kv-centric-round11",image=image)
CANDIDATE_IDS=(0,1,2)


@app.function(gpu="B200",timeout=8 * 60,volumes={str(REMOTE/"build"):CACHE},max_containers=1)
def experiment(mode:str,chosen:list,provenance:dict):
    import os
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    os.chdir(REMOTE);sys.path.insert(0,str(REMOTE))
    started=time.monotonic()
    from api11 import source_evidence
    for name,digest in provenance["source_hashes"].items():
        if hashlib.sha256((REMOTE_BASE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("source mount mismatch: "+name)
    import benchmark11
    try:result=benchmark11.run(mode,chosen)
    except Exception:
        result=dict(benchmark11.CURRENT)
        result.update(passed=False,error=traceback.format_exc())
        print(result["error"],flush=True)
    result.update(source_evidence());result["provenance"]=provenance
    sanitizer=shutil.which("compute-sanitizer")
    if result["passed"] and sanitizer is None:
        result.update(passed=False,sanitizer_error="compute-sanitizer unavailable")
    variants=list(CANDIDATE_IDS) if mode=="candidates" else chosen
    for tool in (("memcheck","synccheck") if result["passed"] else ()):
        remaining=450-(time.monotonic()-started)
        if remaining<10:
            result.update(passed=False,error="insufficient budget for sanitizer; split instead of raising timeout");break
        command=[sanitizer,"--tool",tool,"--error-exitcode","7",sys.executable,"-B","benchmark11.py",
                 "--smoke",",".join(map(str,variants))]
        try:
            check=subprocess.run(command,capture_output=True,text=True,timeout=min(90,remaining))
            result[tool]={"returncode":check.returncode,"stdout":check.stdout,"stderr":check.stderr}
            log=check.stdout+check.stderr
            print(tool+" returncode="+str(check.returncode),flush=True)
            if check.returncode or "ERROR SUMMARY: 0 errors" not in log:
                result["passed"]=False;print(log,flush=True);break
        except subprocess.TimeoutExpired as error:
            result.update(passed=False,error=repr(error));break
    result.update(remote_seconds=time.monotonic()-started,modal_app_id=app.app_id)
    CACHE.commit()
    return result


@app.local_entrypoint()
def main(run_id:int,mode:str="candidates",variants:str="0,1,2"):
    if not 1<=run_id<=3:raise ValueError("planned two GPU calls, at most one corrective spare")
    if mode not in ("candidates","confirmation"):raise ValueError("mode")
    chosen=list(map(int,variants.split(",")))
    if not chosen or len(set(chosen))!=len(chosen) or any(i not in CANDIDATE_IDS for i in chosen):raise ValueError("variants")
    for p in HERE.glob("*.py"):ast.parse(p.read_text(),filename=str(p))
    preserved=json.loads((HERE/"preserved_manifest.json").read_text())["sha256"]
    for n,digest in preserved.items():
        if hashlib.sha256((BASE.parent/n).read_bytes()).hexdigest()!=digest:raise RuntimeError("preserved file changed: "+n)
    paths=[BASE/n for n in BASE_FILES]+[HERE/n for n in NEW_FILES]
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    results=HERE/"results";results.mkdir(exist_ok=True)
    if mode=="confirmation":
        eligible=[]
        for p in results.glob("run_*_candidates.json"):
            data=json.loads(p.read_text());old=data.get("provenance",{}).get("source_hashes",{})
            if data.get("passed") and all(old.get(n)==h for n,h in hashes.items()):eligible.append(p.name)
        if not eligible:raise RuntimeError("source-matched passed candidates gate required")
    ledger=results/f"run_{run_id:03d}_attempt.json"
    record={"status":"reserved","run_id":run_id,"mode":mode,"chosen":chosen,"utc":datetime.now(timezone.utc).isoformat()}
    with ledger.open("x") as f:json.dump(record,f,indent=2)
    provenance={"source_hashes":hashes,"timeout_seconds":8*60,"chosen":chosen,"mode":mode,
                "preserved_files":len(preserved),"run_id":run_id}
    try:
        call=experiment.spawn(mode,chosen,provenance)
        record.update(status="running",call_id=call.object_id)
        ledger.write_text(json.dumps(record,indent=2)+"\n")
        print("MODAL_CALL "+call.object_id+" ledger="+str(ledger),flush=True)
        result=call.get()
    except BaseException as error:
        record.update(status="client_interrupted_remote_state_unknown",error=repr(error))
        ledger.write_text(json.dumps(record,indent=2)+"\n");raise
    target=results/f"run_{run_id:03d}_{mode}.json"
    with target.open("x") as f:json.dump(result,f,indent=2)
    record.update(status="finished",passed=result["passed"],result_file=target.name)
    ledger.write_text(json.dumps(record,indent=2)+"\n")
    print("SAVED "+str(target)+" passed="+str(result["passed"]),flush=True)
    if not result["passed"]:raise RuntimeError("GPU gate failed; inspect saved failure before next call")
