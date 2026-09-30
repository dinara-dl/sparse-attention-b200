"""Round11: token-major Q [B,Sq,Hq,D] and partial O [B,Sq,16,Hq,D] with TMA gather/scatter.

Controls (round9 split8/split8_bf16) keep Q [B,Hq,Sq,D] and partial [B,Hq,Sq,16,D];
`Prepared` takes Q in token-major layout for every variant and makes the head-major
copy for a control once, outside timing. Final O [B,Hq,Sq,D] and LSE [B,Hq,Sq,16]
are shared across variants, so partial/final tensors compare bit for bit through views.
"""
from pathlib import Path
import hashlib
import os
import sys
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
R9=BASE/"optimization_round9"
R8=BASE/"optimization_round8"
R4=BASE/"optimization_round4"
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(R4))
import gpu_plan as old_plan

VARIANTS={"split8":-2,"split8_bf16":-1,"tma8":0,"tma8_bf16":1,"tma16_bf16":2}
CHUNKS={-2:8,-1:8,0:8,1:8,2:16}
PARTIAL_DTYPE={-2:torch.float32,-1:torch.bfloat16,0:torch.float32,1:torch.bfloat16,2:torch.bfloat16}
TOKEN_MAJOR={-2:False,-1:False,0:True,1:True,2:True}
CANDIDATES=(0,1,2)
PARENT_OF={0:-2,1:-1,2:-1}
EXACT=CANDIDATES
_OPS=None


def load_extension():
    global _OPS
    if _OPS is None:
        from torch.utils.cpp_extension import load
        build=HERE/"build";build.mkdir(exist_ok=True)
        os.environ.setdefault("MAX_JOBS","2")
        load(name="xomtention_kvopt11",sources=[str(HERE/"wrapper11.cpp"),str(HERE/"tma11_kernel.cu"),str(HERE/"combine11.cu"),
             str(R9/"split9_kernel.cu"),str(R8/"combine8.cu"),
             str(R4/"gpu_metadata.cu"),str(BASE/"optimization_round1/combine_warp.cu")],
             build_directory=str(build),extra_cflags=["-O2"],
             extra_cuda_cflags=["-O2","-lineinfo","--ptxas-options=-v","-gencode=arch=compute_100a,code=sm_100a"],
             extra_ldflags=["-lcuda"],is_python_module=False,verbose=True)
        _OPS=torch.ops.xomtention_kvopt11
        old_plan.load_extension=lambda:_OPS
    return _OPS


class Prepared:
    """q_tm: [B,Sq,Hq,D] BF16 contiguous. k, v: [B,Hkv,Skv,D]. selected: [B,Hkv,Sq,16] CUDA int32."""
    def __init__(self,q_tm,k,v,selected,variant=-2,workspace=None):
        if variant not in CHUNKS:raise ValueError("variant")
        b,sq,hq,d=q_tm.shape
        if tuple(selected.shape)!=(b,k.shape[1],sq,16):raise ValueError("selected shape")
        if any(t.device!=q_tm.device for t in (k,v,selected)):raise ValueError("device mismatch")
        self.q_tm,self.k,self.v=q_tm,k,v
        self.variant=variant;self.ops=load_extension();self.token_major=TOKEN_MAJOR[variant]
        # Controls consume the head-major layout: one copy at construction, never timed.
        self.q=q_tm if self.token_major else q_tm.permute(0,2,1,3).contiguous()
        self.plan=old_plan.GPUPlan(selected,k.shape[2]//128,hq//k.shape[1],CHUNKS[variant])
        self.tasks,self.qids,self.ranks=self.plan.tasks,self.plan.qids,self.plan.ranks
        dtype=PARTIAL_DTYPE[variant]
        shape=(b,sq,16,hq,d) if self.token_major else (b,hq,sq,16,d)
        if workspace is None:
            self.partial=torch.empty(shape,device=q_tm.device,dtype=dtype)
            self.lses=torch.empty((b,hq,sq,16),device=q_tm.device,dtype=torch.float32)
            self.out=torch.empty((b,hq,sq,d),device=q_tm.device,dtype=torch.bfloat16)
            self.out_lse=torch.empty((b,hq,sq),device=q_tm.device,dtype=torch.float32)
        else:
            if workspace.q_tm.shape!=q_tm.shape or workspace.q_tm.device!=q_tm.device:raise ValueError("workspace")
            self.lses,self.out,self.out_lse=workspace.lses,workspace.out,workspace.out_lse
            # One partial buffer per (dtype, layout) family; a new family allocates its own.
            self.partial=workspace.partial if (workspace.partial.dtype==dtype and tuple(workspace.partial.shape)==shape) \
                else torch.empty(shape,device=q_tm.device,dtype=dtype)

    def partial_view(self):
        """[B,Hq,Sq,16,D] view of the partial buffer for layout-independent comparisons."""
        return self.partial.permute(0,3,1,2,4) if self.token_major else self.partial

    def poison(self):
        for t in (self.partial,self.lses,self.out,self.out_lse):t.fill_(float("nan"))

    def metadata(self):self.plan.build()
    def main(self):
        if self.token_major:self.ops.main_tm_out(self.q,self.k,self.v,self.tasks,self.qids,self.ranks,self.partial,self.lses,self.variant)
        else:self.ops.main_out(self.q,self.k,self.v,self.tasks,self.qids,self.ranks,self.partial,self.lses,self.variant)
    def combine(self):
        if self.token_major:self.ops.combine_tm_out(self.partial,self.lses,self.out,self.out_lse)
        else:self.ops.combine_out(self.partial,self.lses,self.out,self.out_lse)
    def core(self):self.main();self.combine()
    def run(self):self.metadata();self.core()


class CombineAdapter:
    """validate_gpu's combine_unit check: FP32 partial in [B,Hq,Sq,16,D] -> round1 Combine."""
    def combine(self,partial,lses):
        out=torch.empty((*partial.shape[:3],128),device=partial.device,dtype=torch.bfloat16)
        lse=torch.empty(partial.shape[:3],device=partial.device,dtype=torch.float32)
        load_extension().combine_out(partial,lses,out,lse)
        return out,lse


def source_evidence():
    paths=[p for p in BASE.rglob("*") if p.is_file() and p.suffix in (".py",".cu",".cuh",".cpp")
           and "results" not in p.parts and "build" not in p.parts]
    keep={"primitives.cuh","metadata.py","reference.py","api.py","validate_gpu.py",
          "optimization_round4/bench_api4.py","optimization_round4/planning.py",
          "optimization_round4/gpu_plan.py","optimization_round4/timing.py",
          "optimization_round4/gpu_metadata.cu","optimization_round7/timing7.py",
          "optimization_round8/combine8.cu","optimization_round9/split9_kernel.cu",
          "optimization_round1/combine_warp.cu"}
    sources={str(p.relative_to(BASE)):p.read_text() for p in sorted(paths)
             if p.parent==HERE or str(p.relative_to(BASE)) in keep}
    return {"source_text":sources,"source_hashes":{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}}
