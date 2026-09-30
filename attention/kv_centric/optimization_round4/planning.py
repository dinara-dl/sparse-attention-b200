"""Bounded same-KV grouping: preserves every edge, its order and selected rank."""
import torch

CHUNKS={-1:1,0:1,1:4,2:8,3:16,4:1,5:8,6:16,7:8}


def group_tasks(plan,variant):
    if variant not in CHUNKS: raise ValueError("unknown variant")
    limit=128//plan.group_size*CHUNKS[variant]
    ptr=plan.row_ptr.tolist()
    tasks=[]
    for row,(begin,end) in enumerate(zip(ptr,ptr[1:])):
        for start in range(begin,end,limit):
            tasks.append((row//plan.kv_blocks,row%plan.kv_blocks,start,min(limit,end-start)))
    result=torch.tensor(tasks,dtype=torch.int32).reshape(-1,4)
    visits=torch.zeros(plan.query_ids.numel(),dtype=torch.int32)
    for group,block,start,count in result.tolist():
        row=group*plan.kv_blocks+block
        assert 0<count<=limit and ptr[row]<=start<start+count<=ptr[row+1]
        visits[start:start+count]+=1
    if not torch.all(visits==1): raise AssertionError("task edge coverage")
    return result
