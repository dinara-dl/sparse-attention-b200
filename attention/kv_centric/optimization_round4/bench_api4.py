"""Prepared GPU core ablations. CPU CSR/grouping is not hidden in GPU timing."""
from pathlib import Path
import hashlib
import os
import sys
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from metadata import build_plan,validate_plan
from planning import group_tasks,CHUNKS

VARIANTS={"control":-1,"private_sO":0,"resident4":1,"resident8":2,
          "resident16":3,"folded_V":4,"folded_resident8":5,
          "folded_resident16":6,"early_folded_resident8":7}
_OPS=None


def load_extension():
    global _OPS
    if _OPS is None:
        from torch.utils.cpp_extension import load
        build=HERE/"build";build.mkdir(exist_ok=True)
        (build/"ptx").mkdir(exist_ok=True)
        os.environ.setdefault("MAX_JOBS","2")
        load(name="xomtention_kvopt4",sources=[str(HERE/"wrapper.cpp"),
             str(HERE/"resident_kernel.cu"),str(HERE/"gpu_metadata.cu"),str(BASE/"optimization_round3/row_kernel.cu"),
             str(BASE/"optimization_round1/combine_warp.cu")],build_directory=str(build),
             extra_cflags=["-O2"],extra_cuda_cflags=["-O2","-lineinfo","--ptxas-options=-v",
                 "--keep","--keep-dir",str(build/"ptx"),"-gencode=arch=compute_100a,code=sm_100a"],
             extra_ldflags=["-lcuda"],is_python_module=False,verbose=True)
        _OPS=torch.ops.xomtention_kvopt4
    return _OPS


class Workspace:
    def __init__(self,q,k,selected,plan=None):
        self.plan=plan or build_plan(selected,k.shape[2]//128,q.shape[1]//k.shape[1])
        validate_plan(self.plan,selected)
        p=self.plan
        if (p.batch,p.hkv,p.sq,p.group_size)!=(q.shape[0],k.shape[1],q.shape[2],q.shape[1]//k.shape[1]):
            raise ValueError("plan/QKV shape mismatch")
        self.qshape=tuple(q.shape);self.device=q.device
        self.qids=p.query_ids.to(q.device);self.ranks=p.ranks.to(q.device)
        self.partial=torch.empty((*q.shape[:-1],16,128),device=q.device,dtype=torch.float32)
        self.lses=torch.empty((*q.shape[:-1],16),device=q.device,dtype=torch.float32)
        self.out=torch.empty_like(q)
        self.out_lse=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)


class Prepared:
    def __init__(self,q,k,v,selected,*,variant=-1,plan=None,workspace=None):
        if variant not in CHUNKS:raise ValueError("unknown variant")
        self.ops=load_extension();self.q,self.k,self.v=q,k,v;self.variant=variant
        self.workspace=workspace or Workspace(q,k,selected,plan)
        w=self.workspace
        if w.qshape!=tuple(q.shape) or w.device!=q.device:raise ValueError("workspace shape/device")
        self.plan=w.plan
        self.tasks_cpu=group_tasks(self.plan,variant);self.tasks=self.tasks_cpu.to(q.device)
        self.qids,self.ranks=w.qids,w.ranks
        self.partial,self.lses,self.out,self.out_lse=w.partial,w.lses,w.out,w.out_lse

    def poison(self):
        for t in (self.partial,self.lses,self.out,self.out_lse):t.fill_(float("nan"))

    def main(self):
        self.ops.main_out(self.q,self.k,self.v,self.tasks,self.qids,self.ranks,
                          self.partial,self.lses,self.variant)

    def combine(self):
        self.ops.combine_out(self.partial,self.lses,self.out,self.out_lse)

    def run(self):
        self.main();self.combine()


def correctness_adapter(variant,diffs):
    def attention(q,k,v,selected,plan=None,return_debug=False):
        p=Prepared(q,k,v,selected,variant=variant,plan=plan)
        p.poison();p.run()
        if variant!=-1:
            c=Prepared(q,k,v,selected,variant=-1,plan=p.plan)
            c.poison();c.run()
            record={"partial_equal":torch.equal(p.partial,c.partial),
                    "partial_lse_equal":torch.equal(p.lses,c.lses),
                    "output_equal":torch.equal(p.out,c.out),
                    "output_lse_equal":torch.equal(p.out_lse,c.out_lse)}
            record["partial_max_abs"]=float((p.partial-c.partial).abs().max())
            record["partial_lse_max_abs"]=float((p.lses-c.lses).abs().max())
            diffs.append(record)
            if not all(record[key] for key in ("partial_equal","partial_lse_equal","output_equal","output_lse_equal")):
                raise AssertionError(f"strict round3 equivalence failed: {record}")
        return (p.out,p.out_lse,p.partial,p.lses) if return_debug else p.out
    return attention


class ControlCombine:
    def combine(self,partial,lses):
        out=torch.empty((*partial.shape[:3],128),device=partial.device,dtype=torch.bfloat16)
        lse=torch.empty(partial.shape[:3],device=partial.device,dtype=torch.float32)
        load_extension().combine_out(partial,lses,out,lse)
        return out,lse


def source_evidence():
    paths=[BASE/name for name in ("primitives.cuh","metadata.py","reference.py","api.py","validate_gpu.py",
                                "optimization_round3/row_kernel.cu","optimization_round1/combine_warp.cu")]
    paths.extend(p for p in HERE.iterdir() if p.suffix in (".py",".cu",".cpp",".cuh"))
    sources={str(p.relative_to(BASE)):p.read_text() for p in sorted(paths)}
    return {"source_text":sources,"source_hashes":{name:hashlib.sha256(s.encode()).hexdigest() for name,s in sources.items()}}
