"""Round11 gates then timing. Every tma candidate must match its round9 parent bit
for bit (tma8 <-> split8, tma*_bf16 <-> split8_bf16): only layouts and the copy
mechanism change; partials compare through a [B,Hq,Sq,16,D] view. Q is generated
token-major [B,Sq,Hq,D]; the oracle and the controls see head-major copies made
outside timing. Timing protocol: round7 harness."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import torch
from api11 import HERE,BASE,VARIANTS,CANDIDATES,EXACT,PARENT_OF,PARTIAL_DTYPE,TOKEN_MAJOR,Prepared,CombineAdapter,load_extension,source_evidence
from metadata import make_selected
import validate_gpu as suite
sys.path.insert(0,str(BASE/"optimization_round7"))
from timing7 import graph_timings

CURRENT={"passed":False,"correctness":{},"cases":[]}
CONTROLS=(("split8",-2),("split8_bf16",-1))


def exact(a,b):
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def compare_to_parent(p,c,variant):
    """Bit-identical partial O/LSE and final O/LSE against the candidate's parent."""
    exact(p.lses,c.lses);exact(p.out_lse,c.out_lse);exact(p.partial_view(),c.partial_view());exact(p.out,c.out)
    return {"partial":0.0,"lses":0.0,"out":0.0,"out_lse":0.0}


def correctness(variant,smoke=False):
    comparisons=[]
    def attention(q,k,v,selected,plan=None,return_debug=False):
        live=selected.to(q.device);q_tm=q.permute(0,2,1,3).contiguous()   # suite generates [B,Hq,Sq,D]
        p=Prepared(q_tm,k,v,live,variant);p.poison();p.run()
        audit=p.plan.audit(selected)
        c=Prepared(q_tm,k,v,live,PARENT_OF[variant]);c.poison();c.run()
        comparisons.append({"audit":audit,"max_abs":compare_to_parent(p,c,variant)})
        return (p.out,p.out_lse,p.partial_view(),p.lses) if return_debug else p.out
    suite.attention=attention;suite.load_extension=lambda:CombineAdapter()
    result=suite.run_suite(smoke=smoke)
    result["resident_tail"]=suite.run_case("resident_tail",1,1,16,137,4096,"identical",5201,None,True)
    result.update(variant=variant,comparisons=comparisons,
                  exact_parent=next(n for n,v in VARIANTS.items() if v==PARENT_OF[variant]),gate="bit-exact",
                  partial_dtype=str(PARTIAL_DTYPE[variant]),metadata="GPU CSR/tasks from live CUDA IDs")
    return result


def dynamic_graph_check(variant):
    torch.manual_seed(5207)
    q=torch.randn((1,16,137,128),device="cuda",dtype=torch.bfloat16)
    k=torch.randn((1,1,4096,128),device="cuda",dtype=torch.bfloat16);v=torch.randn_like(k)
    ids0=make_selected(1,1,137,32,"identical",5207)
    ids1=make_selected(1,1,137,32,"independent",5209)
    q=q.permute(0,2,1,3).contiguous()   # token-major [B,Sq,Hq,D]
    live=ids0.to(q.device);p=Prepared(q,k,v,live,variant);p.run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):p.run()
    checks=[]
    for ids in (ids1,ids0,ids1):
        live.copy_(ids);q.mul_(-0.75);v.add_(0.03125)
        p.poison();graph.replay()
        audit=p.plan.audit(ids)
        c=Prepared(q,k,v,live,PARENT_OF[variant]);c.run()
        audit["max_abs"]=compare_to_parent(p,c,variant)
        checks.append(audit)
    return checks


def spot_reference(p,ids):
    q,k,v=p.q_tm.permute(0,2,1,3),p.k,p.v;b,hq,sq,_=q.shape;g=hq//k.shape[1]
    errors=[];offset=torch.arange(128,device=q.device)
    for bi,h,qi in sorted({(0,0,0),(b-1,hq-1,sq-1),(0,hq//2,sq//2),(0,min(hq-1,1),min(sq-1,127))}):
        tokens=(ids[bi,h//g,qi].to(device=q.device,dtype=torch.long)[:,None]*128+offset).flatten()
        scores=q[bi,h,qi].float()@k[bi,h//g,tokens].float().T/math.sqrt(128)
        expected=torch.softmax(scores,-1)@v[bi,h//g,tokens].float()
        errors.append({"row":[bi,h,qi],"o":suite.checked(p.out[bi,h,qi],expected),
                       "lse":suite.checked(p.out_lse[bi,h,qi],torch.logsumexp(scores,-1),lse=True)})
    return errors


def gpu_state():
    cmd=["nvidia-smi","--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory","--format=csv,noheader"]
    return subprocess.run(cmd,capture_output=True,text=True,check=True).stdout.strip()


def bench_case(spec,variants,confirmation=False):
    name,b,hkv,g,sq,skv,pattern,seed=spec
    print("CASE "+str(spec),flush=True)
    torch.manual_seed(seed)
    q=torch.randn((b,sq,hkv*g,128),device="cuda",dtype=torch.bfloat16)   # token-major
    k=torch.randn((b,hkv,skv,128),device="cuda",dtype=torch.bfloat16);v=torch.randn_like(k)
    ids=make_selected(b,hkv,sq,skv//128,pattern,seed);live=ids.to(q.device)
    functions,before,checks,prepared={},{},{},{}
    workspace=None;snapshots={}
    # Confirmation drops `core` (operator - metadata) to fit 11 shapes in one call.
    stages=("main","operator") if confirmation else ("main","core","operator")
    for label,variant in variants:
        p=Prepared(q,k,v,live,variant,workspace);workspace=workspace or p
        p.poison();p.run();torch.cuda.synchronize()
        audit=p.plan.audit(ids)
        if not torch.isfinite(p.partial.float()).all().item() or not torch.isfinite(p.lses).all().item():
            raise AssertionError("unwritten partials")
        # One snapshot per partial dtype: FP32 variants agree bit for bit among
        # themselves, BF16 variants among themselves; LSE agrees across all.
        key=str(p.partial.dtype)
        if key not in snapshots:snapshots[key]=(p.out.clone(),p.out_lse.clone(),p.partial_view().clone())
        diff=verify_against_snapshot(p,snapshots[key],variant)
        checks[label]={"audit":audit,"reference":spot_reference(p,ids),"final_vs_first_same_dtype":diff}
        prepared[label]=p
        for stage in stages:
            fn={"main":p.main,"core":p.core,"operator":p.run}[stage]
            functions[label+"/"+stage]=fn
            if stage!="operator":before[label+"/"+stage]=p.metadata
        # The BF16 Combine kernel is shared by every BF16 variant; time it per
        # variant while tuning, once (split8_bf16) in confirmation.
        # Combine differs per layout and dtype: time it for every variant.
        functions[label+"/combine"]=p.combine;before[label+"/combine"]=p.run
    functions["control/metadata"]=workspace.metadata
    trailing=tuple(k for k in functions if k.endswith("/combine") or k=="control/metadata")
    state=gpu_state()
    timing=graph_timings(functions,before,rounds=5,target_ms=100 if confirmation else 60,
                         warmup_ms=150,settle_ms=60,trailing=trailing)
    for label,p in prepared.items():
        p.poison();p.run();verify_against_snapshot(p,snapshots[str(p.partial.dtype)],p.variant)
    flops=4*b*hkv*g*sq*16*128*128
    for key,m in timing["measurements"].items():
        if key.endswith(("/core","/operator")):m["useful_sparse_tflops"]=flops/(m["median_ms"]*1e9)
    record={"name":name,"shape":{"B":b,"Hq":hkv*g,"Hkv":hkv,"Sq":sq,"Skv":skv,"D":128,"TopK":16},
            "seed":seed,"pattern":pattern,"dtype":"BF16","partial_dtype":{l:str(PARTIAL_DTYPE[v]) for l,v in variants},
            "causal":False,"selected_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),"variants":variants,
            "same_inputs":True,"q_layout":{l:("[B,Sq,Hq,D]" if TOKEN_MAJOR[v] else "[B,Hq,Sq,D] copy made at construction") for l,v in variants},"shared_output_workspace":"out/out_lse/lses shared; one partial buffer per (dtype,layout) family",
            "validation":checks,"timings":timing,"useful_qk_pv_flops":flops,"gpu_before":state,"gpu_after":gpu_state(),
            "timed_regions":{"core":"prepared GPU routes -> main + Combine of that variant",
                             "operator":"live CUDA IDs -> GPU CSR/tasks -> main -> Combine of that variant"}}
    print("SUMMARY "+json.dumps({k:round(m["median_ms"],6) for k,m in timing["measurements"].items()}),flush=True)
    return record


def verify_against_snapshot(p,snapshot,variant):
    exact(p.out_lse,snapshot[1]);exact(p.out,snapshot[0]);exact(p.partial_view(),snapshot[2])
    return {"out_max_abs":0.0,"partial_max_abs":0.0,"out_lse":0.0}


def specs(confirmation):
    if not confirmation:
        return [("large_g16",8,1,16,4096,4096,"independent",5301),
                ("g4",1,2,4,4096,32768,"independent",5302),
                ("short_g16",1,1,16,129,32768,"independent",5303)]
    return [("large_g16_seed1",8,1,16,4096,4096,"independent",5901),
            ("large_g16_seed2",8,1,16,4096,4096,"independent",5911),
            ("large_g16_seed3",8,1,16,4096,4096,"independent",5921),
            ("shared",1,1,16,4096,32768,"tile_shared",5902),
            ("hot",1,1,16,4096,32768,"hot",5903),
            ("identical",1,1,16,4096,32768,"identical",5904),
            ("q8k",1,1,16,8192,8192,"independent",5905),
            ("g1",1,2,1,4096,32768,"independent",5906),
            ("g4",1,2,4,4096,32768,"independent",5907),
            ("short_g16",1,1,16,129,32768,"independent",5908),
            ("short_g1",1,1,1,33,32768,"independent",5909)]


def artifacts():
    binary=HERE/"build/xomtention_kvopt11.so"
    result={"resources":subprocess.run(["cuobjdump","--dump-resource-usage",str(binary)],capture_output=True,text=True,check=True,timeout=30).stdout}
    sass=subprocess.run(["cuobjdump","--dump-sass",str(binary)],capture_output=True,text=True,check=True,timeout=60).stdout
    parts=re.split(r'\n\s*Function : (\S+)\n',sass);histogram={}
    for i in range(1,len(parts),2):
        name,body=parts[i],parts[i+1]
        if "main_kernel" not in name and "combine" not in name:continue
        ops=re.findall(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\d\s+)?([A-Z][A-Z0-9_.]*)',body)
        counts={}
        for op in ops:counts[op]=counts.get(op,0)+1
        histogram[name]={"instructions":len(ops),"opcodes":dict(sorted(counts.items(),key=lambda kv:-kv[1]))}
    result["sass_opcode_histogram"]=histogram
    return result


def run(mode,chosen):
    started=time.monotonic();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    load_extension();CURRENT.update(mode=mode,device=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda)
    candidate_ids=list(CANDIDATES) if mode=="candidates" else chosen
    variants=list(CONTROLS)
    variants+=[(next(n for n,v in VARIANTS.items() if v==i),i) for i in candidate_ids]
    CURRENT["variants"]=variants
    for label,variant in variants:
        if variant<0:continue
        if time.monotonic()-started>300:raise RuntimeError("split the remaining work; keep timeout 8*60")
        CURRENT["correctness"][label]=correctness(variant)
        CURRENT["correctness"][label]["dynamic_graph"]=dynamic_graph_check(variant)
        print("CORRECTNESS "+label+" passed",flush=True)
    for spec in specs(mode=="confirmation"):
        if time.monotonic()-started>345:raise RuntimeError("split remaining benchmark, never raise timeout")
        CURRENT["cases"].append(bench_case(spec,variants,mode=="confirmation"))
        gc.collect();torch.cuda.empty_cache()
    CURRENT.update(artifacts());CURRENT.update(source_evidence())
    CURRENT.update(passed=True,elapsed_seconds=time.monotonic()-started)
    return CURRENT


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",type=str,required=True);args=parser.parse_args()
    torch.set_num_threads(1);load_extension()
    for variant in map(int,args.smoke.split(",")):
        print(json.dumps(correctness(variant,smoke=True)),flush=True)
