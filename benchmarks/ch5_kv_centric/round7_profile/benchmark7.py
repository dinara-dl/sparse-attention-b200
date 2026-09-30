"""Round7: where does rawmax8's time go? Two independent answers in one GPU call.

1. Subtractive attribution: diagnostic builds of rawmax8 with one component
   removed, timed with the round4 protocol (harness fix in timing7.py).
2. Nsight Compute `--set full` on rawmax8 main and Combine, if the container
   grants performance-counter access; otherwise the failure is recorded and
   the attribution still stands. No new correctness candidate exists here.
"""
import argparse
import gc
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import torch
from api7 import HERE,BASE,VARIANTS,DIAGNOSTIC,Prepared,load_extension,source_evidence
from metadata import make_selected
from timing7 import graph_timings

CURRENT={"passed":False,"cases":[],"profile":{}}
PROFILE_SHAPE=(2,1,16,4096,4096,"independent",5501)   # 1 GiB of partials: cheap kernel replay


def gpu_state():
    cmd=["nvidia-smi","--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory","--format=csv,noheader"]
    return subprocess.run(cmd,capture_output=True,text=True,check=True).stdout.strip()


def exact(a,b):
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def attribution_case(spec):
    name,b,hkv,g,sq,skv,pattern,seed=spec
    print("CASE "+str(spec),flush=True)
    torch.manual_seed(seed)
    q=torch.randn((b,hkv*g,sq,128),device="cuda",dtype=torch.bfloat16)
    k=torch.randn((b,hkv,skv,128),device="cuda",dtype=torch.bfloat16);v=torch.randn_like(k)
    ids=make_selected(b,hkv,sq,skv//128,pattern,seed);live=ids.to(q.device)
    parent=Prepared(q,k,v,live,VARIANTS["overlap8"]);parent.poison();parent.run()
    ref=Prepared(q,k,v,live,VARIANTS["rawmax8"],parent);ref.poison();ref.run();torch.cuda.synchronize()
    audit=ref.plan.audit(ids)
    if not torch.isfinite(ref.partial).all().item() or not torch.isfinite(ref.lses).all().item():
        raise AssertionError("rawmax8 left unwritten partials")
    snapshot=(ref.out.clone(),ref.out_lse.clone(),ref.partial.clone(),ref.lses.clone())
    parent.poison();parent.run();exact(parent.out,snapshot[0]);exact(parent.out_lse,snapshot[1])
    exact(parent.partial,snapshot[2]);exact(parent.lses,snapshot[3])
    diag0=Prepared(q,k,v,live,VARIANTS["diag0"],parent);diag0.poison();diag0.run()
    exact(diag0.partial,snapshot[2]);exact(diag0.lses,snapshot[3]);exact(diag0.out,snapshot[0])
    checks={"rawmax8_equals_overlap8":True,"diag0_equals_rawmax8":True,"audit":audit}
    functions,before={},{}
    functions["overlap8/main"]=parent.main;before["overlap8/main"]=parent.metadata
    for stage,fn in (("main",ref.main),("core",ref.core),("operator",ref.run)):
        functions["rawmax8/"+stage]=fn
        if stage!="operator":before["rawmax8/"+stage]=ref.metadata
    prepared={}
    for label,variant in VARIANTS.items():
        if variant<100 or label=="diag0":continue
        p=Prepared(q,k,v,live,variant,parent);prepared[label]=p
        p.run();torch.cuda.synchronize()    # routes built; garbage partials by design
        functions[label+"/main"]=p.main;before[label+"/main"]=p.metadata
    functions["diag0/main"]=diag0.main;before["diag0/main"]=diag0.metadata
    functions["shared/combine"]=parent.combine;before["shared/combine"]=ref.run
    functions["control/metadata"]=parent.metadata
    state=gpu_state()
    timing=graph_timings(functions,before,rounds=5,target_ms=100,warmup_ms=150,settle_ms=60,
                         trailing=("shared/combine","control/metadata"))
    ref.poison();ref.run();exact(ref.out,snapshot[0]);exact(ref.out_lse,snapshot[1])
    flops=4*b*hkv*g*sq*16*128*128
    for key,m in timing["measurements"].items():
        if key.endswith(("/core","/operator")):m["useful_sparse_tflops"]=flops/(m["median_ms"]*1e9)
    partial_bytes=b*hkv*g*sq*16*128*4;q_gather_bytes=b*hkv*g*sq*16*128*2
    record={"name":name,"shape":{"B":b,"Hq":hkv*g,"Hkv":hkv,"Sq":sq,"Skv":skv,"D":128,"TopK":16},
            "seed":seed,"pattern":pattern,"dtype":"BF16","partial_dtype":"FP32",
            "selected_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "validation":checks,"timings":timing,"useful_qk_pv_flops":flops,
            "bytes":{"partial_o_fp32":partial_bytes,"partial_lse":partial_bytes//128,
                     "q_gather_bf16":q_gather_bytes,"kv_bf16":2*b*hkv*skv*128*2},
            "gpu_before":state,"gpu_after":gpu_state(),
            "diagnostic_note":"*/main of diagnostic variants produce garbage partials; only their time is meaningful"}
    print("SUMMARY "+json.dumps({k:round(m["median_ms"],6) for k,m in timing["measurements"].items()}),flush=True)
    return record


def find_ncu():
    found=shutil.which("ncu")
    if found:return found
    for pattern in ("/usr/local/cuda/bin/ncu","/opt/nvidia/nsight-compute/*/ncu","/usr/local/cuda-*/bin/ncu"):
        hits=sorted(glob.glob(pattern))
        if hits:return hits[-1]
    return None


def profile_target():
    """Child process under ncu: two warm runs, then the profiled run."""
    b,hkv,g,sq,skv,pattern,seed=PROFILE_SHAPE
    torch.manual_seed(seed)
    q=torch.randn((b,hkv*g,sq,128),device="cuda",dtype=torch.bfloat16)
    k=torch.randn((b,hkv,skv,128),device="cuda",dtype=torch.bfloat16);v=torch.randn_like(k)
    live=make_selected(b,hkv,sq,skv//128,pattern,seed).to(q.device)
    p=Prepared(q,k,v,live,VARIANTS["rawmax8"])
    for _ in range(2):p.run()
    torch.cuda.synchronize()
    p.run();torch.cuda.synchronize()
    print("PROFILE_TARGET_DONE",flush=True)


def profile(budget_seconds):
    ncu=find_ncu()
    result={"tool":ncu,"shape":PROFILE_SHAPE,"variant":"rawmax8","kernels":"main_kernel, kv_combine_warp"}
    if ncu is None:
        result["error"]="ncu not found in image";print("NCU missing",flush=True);return result
    report=str(HERE/"build"/"rawmax8_profile")
    command=[ncu,"--target-processes","all","--kernel-name-base","demangled",
             "-k","regex:main_kernel|kv_combine_warp","--launch-skip","4","--launch-count","2",
             # Call 1 failed at "Failed to lock GPU clock frequencies": the
             # container may not pin clocks. Profile at whatever clocks the
             # GPU runs, which is also how every benchmark here is timed.
             "--clock-control","none",
             "--set","full","--import-source","no","-f","-o",report,
             sys.executable,"-B",str(HERE/"benchmark7.py"),"--profile-target"]
    result["command"]=command
    # Profiling gets at most 240 s and leaves 90 s for the CSV dumps; the
    # remote function keeps timeout=8*60 whatever happens here.
    started=time.monotonic();limit=max(30,min(240,budget_seconds-90))
    try:
        run=subprocess.run(command,capture_output=True,text=True,timeout=limit)
    except subprocess.TimeoutExpired as error:
        result["error"]="ncu timed out after "+str(limit)+" s";print(result["error"],flush=True);return result
    result.update(returncode=run.returncode,stdout=run.stdout[-20000:],stderr=run.stderr[-20000:],
                  profile_seconds=time.monotonic()-started)
    print("NCU returncode="+str(run.returncode),flush=True)
    if run.returncode or not Path(report+".ncu-rep").exists():
        result["error"]="ncu failed (see stderr; ERR_NVGPUCTRPERM means no counter access)"
        print(run.stderr[-3000:],flush=True);return result
    for page in ("details","raw"):
        try:
            dump=subprocess.run([ncu,"-i",report+".ncu-rep","--csv","--page",page],capture_output=True,text=True,timeout=40)
            result["csv_"+page]=dump.stdout;result["csv_"+page+"_stderr"]=dump.stderr[-4000:]
        except subprocess.TimeoutExpired:
            result["csv_"+page+"_error"]="dump timed out";break
    if "csv_details" not in result:return result
    lines=[l for l in result["csv_details"].splitlines() if any(t in l for t in
           ('"Duration"','"Memory Throughput"','"DRAM Throughput"','"Compute (SM) Throughput"',
            '"Achieved Occupancy"','"Registers Per Thread"','"L2 Hit Rate"','"Issue Slots Busy"'))]
    for l in lines[:40]:print("NCU "+l,flush=True)
    result["ok"]=True
    return result


def artifacts():
    binary=HERE/"build/xomtention_kvopt7.so"
    result={"resources":subprocess.run(["cuobjdump","--dump-resource-usage",str(binary)],capture_output=True,text=True,check=True,timeout=30).stdout}
    sass=subprocess.run(["cuobjdump","--dump-sass",str(binary)],capture_output=True,text=True,check=True,timeout=60).stdout
    parts=re.split(r'\n\s*Function : (\S+)\n',sass);histogram={}
    for i in range(1,len(parts),2):
        name,body=parts[i],parts[i+1]
        if "main_kernel" not in name and "kv_combine_warp" not in name:continue
        ops=re.findall(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\d\s+)?([A-Z][A-Z0-9_.]*)',body)
        counts={}
        for op in ops:counts[op]=counts.get(op,0)+1
        histogram[name]={"instructions":len(ops),"opcodes":dict(sorted(counts.items(),key=lambda kv:-kv[1]))}
    result["sass_opcode_histogram"]=histogram
    return result


def run(mode):
    started=time.monotonic();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    load_extension();CURRENT.update(mode=mode,device=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda)
    CURRENT["variants"]=VARIANTS
    specs=() if mode=="ncu_only" else (("large_g16",8,1,16,4096,4096,"independent",5501),("q8k",1,1,16,8192,8192,"independent",5505))
    for spec in specs:
        if time.monotonic()-started>300:raise RuntimeError("split remaining attribution, never raise timeout")
        CURRENT["cases"].append(attribution_case(spec))
        gc.collect();torch.cuda.empty_cache()
    remaining=440-(time.monotonic()-started)
    CURRENT["profile"]=profile(remaining) if remaining>150 else {"error":"no budget left for ncu; split instead of raising timeout"}
    CURRENT.update(artifacts());CURRENT.update(source_evidence())
    CURRENT.update(passed=True,elapsed_seconds=time.monotonic()-started)
    return CURRENT


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--profile-target",action="store_true");args=parser.parse_args()
    if args.profile_target:
        torch.set_num_threads(1);load_extension();profile_target()
