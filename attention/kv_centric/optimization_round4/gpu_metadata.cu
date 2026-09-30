// GPU reverse CSR and bounded task generation. No host task-count readback.
// Histogram/scatter use CTA-local bins and reserve global ranges per bin.
#include "../primitives.cuh"
#include <cub/device/device_scan.cuh>
#include <algorithm>

namespace kv4_metadata {
constexpr int THREADS=256, ITEMS=4, EDGES_PER_CTA=THREADS*ITEMS;

template<bool SCATTER>
__global__ void edge_bins(const int32_t* ids,int32_t* counts,const int32_t* row_ptr,
    int32_t* cursors,int32_t* qids,int32_t* ranks,int32_t* error,
    int group_edges,int kv_blocks) {
  const int group=blockIdx.y,tid=threadIdx.x;
  extern __shared__ int scratch[];
  int* bins=scratch;int* bases=scratch+kv_blocks;
  for(int bin=tid;bin<kv_blocks;bin+=THREADS)bins[bin]=0;
  __syncthreads();
  int block[ITEMS],position[ITEMS],edge[ITEMS];
  #pragma unroll
  for(int item=0;item<ITEMS;++item) {
    const int e=int(blockIdx.x)*EDGES_PER_CTA+tid+item*THREADS;
    edge[item]=e;block[item]=-1;position[item]=0;
    const unsigned mask=__ballot_sync(0xffffffffu,e<group_edges);
    if(e<group_edges) {
      const int key=ids[int64_t(group)*group_edges+e];
      // A warp covers exactly two 16-ID query lists. Check duplicate IDs
      // within the same half-warp, not across the two independent queries.
      const unsigned peers=__match_any_sync(mask,key);
      const unsigned half=(tid&16)?0xffff0000u:0x0000ffffu;
      if(!SCATTER && __popc(peers&half)>1)atomicOr(error,2);
      if(unsigned(key)<unsigned(kv_blocks)) {
        block[item]=key;position[item]=atomicAdd(bins+key,1);
      } else if(!SCATTER)atomicOr(error,1);
    }
  }
  __syncthreads();
  for(int bin=tid;bin<kv_blocks;bin+=THREADS) {
    const int n=bins[bin],row=group*kv_blocks+bin;
    if constexpr(SCATTER) {
      bases[bin]=n?row_ptr[row]+atomicAdd(cursors+row,n):0;
    } else if(n)atomicAdd(counts+row,n);
  }
  if constexpr(SCATTER) {
    __syncthreads();
    #pragma unroll
    for(int item=0;item<ITEMS;++item) {
      if(block[item]>=0) {
        const int dst=bases[block[item]]+position[item];
        qids[dst]=edge[item]/16;ranks[dst]=edge[item]%16;
      }
    }
  }
}

__global__ void chunk_counts(const int32_t* counts,int32_t* chunks,
    const int32_t* error,int rows,int capacity) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row<=rows)chunks[row]=(row<rows && !*error)?(counts[row]+capacity-1)/capacity:0;
}

__global__ void make_tasks(const int32_t* row_ptr,const int32_t* task_ptr,
    int32_t* tasks,int rows,int kv_blocks,int capacity) {
  const int row=blockIdx.x;
  if(row>=rows)return;
  const int start=row_ptr[row],end=row_ptr[row+1];
  for(int task=task_ptr[row]+threadIdx.x;task<task_ptr[row+1];task+=blockDim.x) {
    const int begin=start+(task-task_ptr[row])*capacity;
    int4 value={row/kv_blocks,row%kv_blocks,begin,min(capacity,end-begin)};
    *reinterpret_cast<int4*>(tasks+int64_t(task)*4)=value;
  }
}
}

int64_t kv4_metadata_temp_bytes(int rows) {
  size_t bytes=0;
  kv0::cuda_check(cub::DeviceScan::ExclusiveSum(nullptr,bytes,
      static_cast<const int32_t*>(nullptr),static_cast<int32_t*>(nullptr),rows+1));
  return int64_t(bytes);
}

void kv4_build_metadata(const int32_t* selected,int32_t* counts,int32_t* row_ptr,
    int32_t* cursors,int32_t* qids,int32_t* ranks,int32_t* chunk_counts,
    int32_t* task_ptr,int32_t* tasks,int32_t* error,void* temp,size_t temp_bytes,
    int groups,int sq,int kv_blocks,int g,int chunks_per_cta,int max_tasks,cudaStream_t stream) {
  const int rows=groups*kv_blocks,capacity=128/g*chunks_per_cta;
  kv0::cuda_check(cudaMemsetAsync(counts,0,size_t(rows+1)*4,stream));
  kv0::cuda_check(cudaMemsetAsync(cursors,0,size_t(rows)*4,stream));
  kv0::cuda_check(cudaMemsetAsync(error,0,4,stream));
  // Invalid padding CTAs see count=0 and exit before allocation/barriers.
  kv0::cuda_check(cudaMemsetAsync(tasks,0,size_t(max_tasks)*4*4,stream));
  // The rounded-up count can exceed INT_MAX even though every valid edge
  // index fits int32. Do the launch arithmetic in int64 before narrowing.
  dim3 grid(unsigned((int64_t(sq)*16+kv4_metadata::EDGES_PER_CTA-1)/
                     kv4_metadata::EDGES_PER_CTA),unsigned(groups));
  const int shared_bytes=2*kv_blocks*int(sizeof(int));
  kv4_metadata::edge_bins<false><<<grid,256,shared_bytes,stream>>>(
      selected,counts,row_ptr,cursors,qids,ranks,error,sq*16,kv_blocks);
  kv0::cuda_check(cudaGetLastError());
  kv0::cuda_check(cub::DeviceScan::ExclusiveSum(temp,temp_bytes,counts,row_ptr,rows+1,stream));
  kv4_metadata::edge_bins<true><<<grid,256,shared_bytes,stream>>>(
      selected,counts,row_ptr,cursors,qids,ranks,error,sq*16,kv_blocks);
  kv0::cuda_check(cudaGetLastError());
  kv4_metadata::chunk_counts<<<(rows+256)/256,256,0,stream>>>(counts,chunk_counts,error,rows,capacity);
  kv0::cuda_check(cudaGetLastError());
  kv0::cuda_check(cub::DeviceScan::ExclusiveSum(temp,temp_bytes,chunk_counts,task_ptr,rows+1,stream));
  kv4_metadata::make_tasks<<<rows,128,0,stream>>>(row_ptr,task_ptr,tasks,rows,kv_blocks,capacity);
  kv0::cuda_check(cudaGetLastError());
}
