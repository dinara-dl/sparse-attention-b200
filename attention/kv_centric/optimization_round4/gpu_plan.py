"""Experimental entirely-device routing. Not promoted before B200 validation.

Construction allocates buffers for fixed geometry; build/run read live CUDA IDs.
No .item(), .cpu(), host sort, or host task-count read occurs in build/run.
audit() is deliberately synchronous and only for correctness tests.
"""
import torch
from bench_api4 import load_extension
from planning import CHUNKS


def task_bound(edges,rows,g,chunk_size):
    if min(edges,rows)>0 and g in (1,2,4,8,16) and chunk_size in (1,4,8,16):
        capacity=128//g*chunk_size
        return min(edges,(edges+capacity-1)//capacity+rows)
    raise ValueError("invalid routing geometry")


class GPUPlan:
    def __init__(self,selected,kv_blocks,g,chunk_size):
        if selected.device.type!="cuda" or selected.dtype!=torch.int32 or not selected.is_contiguous():
            raise ValueError("contiguous CUDA int32 selected required")
        if selected.ndim!=4 or selected.shape[-1]!=16 or min(selected.shape)<=0:
            raise ValueError("selected[B,Hkv,Sq,16]")
        if not 16<=kv_blocks<=4096:raise ValueError("initial GPU routing supports 16..4096 KV blocks")
        self.ops=load_extension();self.selected=selected;self.kv_blocks=kv_blocks
        self.g,self.chunk_size=g,chunk_size
        self.b,self.hkv,self.sq,_=selected.shape
        self.rows=self.b*self.hkv*kv_blocks;self.edges=selected.numel()
        self.max_tasks=task_bound(self.edges,self.rows,g,chunk_size)
        if self.b*self.hkv>65535 or max(self.rows+1,self.edges,self.max_tasks)>=2**31:
            raise ValueError("int32/grid-y routing bounds")
        def ints(n):return torch.empty(n,device=selected.device,dtype=torch.int32)
        self.counts=ints(self.rows+1);self.row_ptr=ints(self.rows+1)
        self.cursors=ints(self.rows);self.qids=ints(self.edges);self.ranks=ints(self.edges)
        self.chunks=ints(self.rows+1);self.task_ptr=ints(self.rows+1)
        self.tasks=ints((self.max_tasks,4));self.error=ints(1)
        with torch.cuda.device(selected.device):
            nbytes=self.ops.metadata_temp_bytes(self.rows)
        self.temp=torch.empty(nbytes,device=selected.device,dtype=torch.uint8)

    def build(self):
        self.ops.metadata_out(self.selected,self.counts,self.row_ptr,self.cursors,
            self.qids,self.ranks,self.chunks,self.task_ptr,self.tasks,self.error,self.temp,
            self.kv_blocks,self.g,self.chunk_size)

    def audit(self,selected_cpu):
        from metadata import Plan,validate_plan,validate_selected
        validate_selected(selected_cpu,self.kv_blocks)
        status=int(self.error.item())
        if status:raise ValueError(f"GPU selected validation failed: flag={status} (1=range, 2=duplicate)")
        ptr=self.row_ptr.cpu();task_ptr=self.task_ptr.cpu();tasks=self.tasks.cpu()
        qids=self.qids.cpu();ranks=self.ranks.cpu();counts=self.counts.cpu()
        if int(ptr[-1])!=self.edges:raise AssertionError("GPU histogram lost edges")
        torch.testing.assert_close(ptr[1:]-ptr[:-1],counts[:-1],atol=0,rtol=0)
        actual=int(task_ptr[-1])
        if not 0<actual<=self.max_tasks:raise AssertionError("task bound exceeded")
        if torch.any(tasks[actual:]!=0):raise AssertionError("invalid padding tasks not cleared")
        # Independently split grouped tasks into single-M128 chunks for the old
        # exact edge-coverage oracle. The device build itself never does this.
        canonical=[]
        for group,block,start,count in tasks[:actual].tolist():
            if not 0<count<=128//self.g*self.chunk_size:raise AssertionError("grouped count")
            for delta in range(0,count,128//self.g):
                canonical.append((group,block,start+delta,min(128//self.g,count-delta)))
        plan=Plan(self.b,self.hkv,self.sq,self.kv_blocks,self.g,ptr,qids,ranks,
                  torch.tensor(canonical,dtype=torch.int32))
        validate_plan(plan,selected_cpu)
        return {"actual_tasks":actual,"launch_bound":self.max_tasks,"padding_ctas":self.max_tasks-actual,
                "edges":self.edges,"gpu_error":status,"exact_coverage":True}


class GPUPrepared:
    def __init__(self,q,k,v,selected,variant=-1,workspace=None):
        if variant not in CHUNKS:raise ValueError("variant")
        if q.ndim!=4 or k.ndim!=4 or q.shape[0]!=k.shape[0] or q.shape[1]%k.shape[1]:
            raise ValueError("QKV/GQA geometry")
        if tuple(selected.shape)!=(q.shape[0],k.shape[1],q.shape[2],16):raise ValueError("selected geometry")
        if any(x.device!=q.device for x in (k,v,selected)):raise ValueError("device mismatch")
        self.q,self.k,self.v=q,k,v;self.variant=variant;self.ops=load_extension()
        self.plan=GPUPlan(selected,k.shape[2]//128,q.shape[1]//k.shape[1],CHUNKS[variant])
        p=self.plan
        self.tasks,self.qids,self.ranks=p.tasks,p.qids,p.ranks
        if workspace is None:
            self.partial=torch.empty((*q.shape[:-1],16,128),device=q.device,dtype=torch.float32)
            self.lses=torch.empty((*q.shape[:-1],16),device=q.device,dtype=torch.float32)
            self.out=torch.empty_like(q);self.out_lse=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
        else:
            if tuple(workspace.out.shape)!=tuple(q.shape) or workspace.out.device!=q.device:
                raise ValueError("GPU output workspace shape/device mismatch")
            self.partial,self.lses=workspace.partial,workspace.lses
            self.out,self.out_lse=workspace.out,workspace.out_lse

    def poison(self):
        for t in (self.partial,self.lses,self.out,self.out_lse):t.fill_(float("nan"))

    def metadata(self):self.plan.build()

    def main(self):
        self.ops.main_out(self.q,self.k,self.v,self.tasks,self.qids,self.ranks,self.partial,self.lses,self.variant)

    def combine(self):self.ops.combine_out(self.partial,self.lses,self.out,self.out_lse)

    def core(self):self.main();self.combine()

    def run(self):self.metadata();self.core()
