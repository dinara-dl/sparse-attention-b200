"""Timing methodology inherited unchanged from round3, with preallocated workspaces."""
import math
import statistics
import torch

def percentile(values,f):
    v=sorted(values);pos=(len(v)-1)*f;lo=math.floor(pos);hi=math.ceil(pos)
    return v[lo]*(hi-pos)+v[hi]*(pos-lo) if lo!=hi else v[lo]


def event_time(fn,repetitions):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repetitions): fn()
    b.record();b.synchronize()
    return float(a.elapsed_time(b))/repetitions


def graph_timings(functions,before_measure,*,rounds=5,target_ms=100,warmup_ms=200,settle_ms=0):
    graphs,iterations,settles,samples={},{},{},{key:[] for key in functions}
    def prepare(key):
        if key in before_measure:
            before_measure[key]()
            torch.cuda.synchronize()
    for key,fn in functions.items():
        prepare(key)
        for _ in range(3): fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4): fn()
        prepare(key)
        pilot=max(event_time(graph.replay,3),.001)
        iterations[key]=max(1,min(10000,math.ceil(target_ms/pilot)))
        settles[key]=max(0,min(10000,math.ceil(settle_ms/pilot)))
        prepare(key)
        event_time(graph.replay,max(1,min(10000,math.ceil(warmup_ms/pilot))))
        graphs[key]=graph
    keys=list(functions);orders=[]
    for r in range(rounds):
        order=keys[r%len(keys):]+keys[:r%len(keys)]
        if r%2: order.reverse()
        orders.append(order)
        for key in order:
            prepare(key)
            if settles[key]: event_time(graphs[key].replay,settles[key])
            samples[key].append(event_time(graphs[key].replay,iterations[key])/4)
    return {"method":"CUDA events around CUDA graph replay; 4 calls/graph",
        "combine_layout_seeding":"matching main before capture/pilot/warmup/each timed round; outside events",
        "orders":orders,"replays_per_round":iterations,"target_ms_per_round":target_ms,
        "settle_ms_before_each_sample":settle_ms,
        "warmup_ms_per_function":warmup_ms,
        "measurements":{key:{"median_ms":statistics.median(v),"p10_ms":percentile(v,.1),
            "p90_ms":percentile(v,.9),"samples_ms":v} for key,v in samples.items()}}



