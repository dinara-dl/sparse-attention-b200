// Round9: two independent Q-chunk pipelines per CTA ("split").
// Round7/8: main is latency-bound at 8 warps/SM; row path, barriers and
// dependency chains leave every pipe under 40%. Instead of a producer/
// consumer protocol, the CTA's 8 warps form two GROUPS. Group g (warps
// 4g..4g+3) processes chunks g, g+2, g+4, ... of the task with
//   its own Q buffer, its own TMEM columns (S/P and O), its own staging,
//   its own qk/pv mbarriers and its own named barrier.
// Shared, read-only after load: K and V tiles. Shared hardware: tensor core.
// Inside a group the schedule is round8's (Q prefetch after QK, next QK
// over the epilogue, bulk-copy epilogue), so exactness carries over:
// FP32 partials are bit-identical to bulk8/rawmax8. The groups drift apart
// naturally, so one group's row path overlaps the other's MMA/epilogue.
#include "../primitives.cuh"
#include <math_constants.h>
#include <cfloat>
#include <cmath>
#include <type_traits>

namespace kv_split9_ops {
using namespace kv0;
constexpr int GROUPS=2,GROUP_THREADS=128,GROUP_WARPS=4;
__device__ __forceinline__ void wait_epoch(int address,int phase) {
  asm volatile("{ .reg .pred p; WAIT: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; "
               "@!p bra WAIT; }" :: "r"(address),"r"(phase) : "memory");
}
// Elected lane of the group's first warp issues that group's MMAs.
__device__ __forceinline__ bool group_issuer(int tid,int g) {
  if ((tid >> 5) != GROUP_WARPS*g) return false;
  unsigned leader;
  asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.b32 %0, 1, 0, p; }"
               : "=r"(leader) :: "memory");
  return leader != 0;
}
// Named barrier over one group's 128 threads; id 0 stays __syncthreads.
__device__ __forceinline__ void group_sync(int g) {
  asm volatile("bar.sync %0, %1;" :: "r"(1+g),"r"(GROUP_THREADS) : "memory");
}
__device__ __forceinline__ void load32(int addr,float (&v)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),"=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15]),"=f"(v[16]),"=f"(v[17]),"=f"(v[18]),"=f"(v[19]),"=f"(v[20]),"=f"(v[21]),"=f"(v[22]),"=f"(v[23]),"=f"(v[24]),"=f"(v[25]),"=f"(v[26]),"=f"(v[27]),"=f"(v[28]),"=f"(v[29]),"=f"(v[30]),"=f"(v[31])
               : "r"(addr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void bulk_store(void* dst,int src,int bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :: "l"(dst),"r"(src),"r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_all() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

template<bool BF16P>
struct Staging {
  using T=std::conditional_t<BF16P,nv_bfloat16,float>;
  // FP32 rows are staged and copied in two 64-column halves through one
  // 34-KiB buffer per group; BF16 rows fit in one pass. Row stride 272 B
  // = 68 words = 4 (mod 32): 16-byte stores from 8 consecutive rows cover
  // all 32 banks without a swizzle.
  static constexpr int HALF_ELEMS=BF16P?D:D/2;
  static constexpr int ROW_ELEMS=HALF_ELEMS+(BF16P?8:4);
  static constexpr int ROW_BYTES=ROW_ELEMS*int(sizeof(T));
  static constexpr int COPY_BYTES=HALF_ELEMS*int(sizeof(T));
  static constexpr int PASSES=BF16P?1:2;
  static constexpr int BYTES=M*ROW_BYTES;
  static_assert(ROW_BYTES==272 && COPY_BYTES==256,"one 272-byte staging row per group row");
};
// Dynamic SMEM: Q[2] | K | V | staging[2]
constexpr int Q_OFFSET=0,K_OFFSET=2*TILE_BYTES,V_OFFSET=3*TILE_BYTES,STAGE_OFFSET=4*TILE_BYTES;

template<int MAX_CHUNKS,bool BF16P>
__global__ __launch_bounds__(256) void main_kernel(
    const nv_bfloat16* __restrict__ q,
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap v_map,
    const int32_t* __restrict__ tasks,const int32_t* __restrict__ query_ids,
    const int32_t* __restrict__ ranks,typename Staging<BF16P>::T* __restrict__ partial_o,
    float* __restrict__ partial_lse,int batch,int hq,int hkv,int sq,int skv,
    int group_size,int edge_count) {
  using S=Staging<BF16P>;
  const int tid=threadIdx.x,warp=tid/32;
  const int g=warp/GROUP_WARPS,gtid=tid-g*GROUP_THREADS,gwarp=warp-g*GROUP_WARPS;
  const int group_log=31-__clz(unsigned(group_size)),group_mask=group_size-1;
  const int* task=tasks+int64_t(blockIdx.x)*4;
  const int group=task[0],block=task[1],start=task[2],count=task[3];
  if(group<0 || group>=batch*hkv || block<0 || block>=skv/N || start<0 ||
     count<=0 || count>MAX_CHUNKS*(M>>group_log) || int64_t(start)+count>edge_count) return;
  const int b=group/hkv,kh=group%hkv;
  const int rows_per_chunk=M>>group_log;
  const int chunk_iterations=(count+rows_per_chunk-1)/rows_per_chunk;
  extern __shared__ __align__(1024) unsigned char storage[];
  const int q_addr=smem_address(storage)+Q_OFFSET+g*TILE_BYTES;
  const int k_addr=smem_address(storage)+K_OFFSET,v_addr=smem_address(storage)+V_OFFSET;
  auto* stage=reinterpret_cast<typename S::T*>(storage+STAGE_OFFSET+g*S::BYTES);
  __shared__ int query_map[GROUPS][2][M],rank_map[GROUPS][2][M];
  __shared__ uint64_t kv_ready,qk_done[GROUPS],pv_done[GROUPS];
  __shared__ int tmem_base;
  const int kv_bar=smem_address(&kv_ready);
  const int qk_bar=smem_address(&qk_done[g]),pv_bar=smem_address(&pv_done[g]);
  if(tid==0) {
    barrier_init(kv_bar);
    for(int i=0;i<GROUPS;++i) {barrier_init(smem_address(&qk_done[i]));barrier_init(smem_address(&pv_done[i]));}
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // Both groups' accumulators: 512 columns, the whole TMEM of the SM.
  if(warp==0) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                           :: "r"(smem_address(&tmem_base)) : "memory");
  __syncthreads();
  const int ts=tmem_base+256*g,to=ts+128;
  constexpr uint32_t desc=(1U<<4)|(1U<<7)|(1U<<10)|((N>>3)<<17)|((M>>4)<<24);
  auto issue_qk = [&]() {
    if(group_issuer(tid,g)) {
      for(int panel=0;panel<2;++panel) {
        uint64_t a=kmajor(q_addr)+panel*(M*128/16);
        uint64_t k=kmajor(k_addr)+panel*(N*128/16);
        for(int micro=0;micro<4;++micro)
          mma_ss(ts,a+micro*2,k+micro*2,desc,panel||micro);
      }
      mma_commit(qk_bar);
    }
  };
  // Group-local gather of chunk `iteration` into this group's Q buffer.
  auto gather = [&](int iteration) {
    const int bank=(iteration>>1)&1,offset=iteration*rows_per_chunk;
    const int real=min(rows_per_chunk,count-offset),local_query=gtid>>group_log;
    {
      int qi=-1,r=-1;
      if(local_query<real) {
        qi=query_ids[start+offset+local_query];r=ranks[start+offset+local_query];
        if(qi<0 || qi>=sq || r<0 || r>=TOPK) qi=r=-1;
      }
      query_map[g][bank][gtid]=qi;rank_map[g][bank][gtid]=r;
    }
    group_sync(g);
    for(int vector=gtid;vector<M*D/8;vector+=GROUP_THREADS) {
      const int row=vector/16,col=(vector%16)*8;
      const int qi=query_map[g][bank][row],head=kh*group_size+(row&group_mask);
      const int offset=(col/64)*M*64+row*64+((col%64)^((row%8)*8));
      const auto* src=q+((int64_t(b)*hq+head)*sq+(qi>=0?qi:0))*D+col;
      const int ignore=qi<0;
      asm volatile("{ .reg .pred zero; setp.ne.b32 zero, %2, 0; "
                   "cp.async.cg.shared.global [%0], [%1], 16, zero; }"
                   :: "r"(q_addr+offset*2),"l"(src),"r"(ignore) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  auto finish_gather = [&]() {
    asm volatile("cp.async.wait_group 0; fence.proxy.async.shared::cta;" ::: "memory");
    group_sync(g);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  };
  auto issue_kv = [&]() {
    if(tid==0) {
      const int kv_row=group*skv+block*N;
      for(int panel=0;panel<2;++panel)
        tma_copy(k_addr+panel*M*128,&k_map,panel*64,kv_row,kv_bar);
      asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
                   "[%0], [%1, {%2,%3,%4,%5}], [%6];"
                   :: "r"(v_addr),"l"(&v_map),"r"(0),"r"(0),"r"(0),
                      "r"(kv_row/8),"r"(kv_bar) : "memory");
      expect_bytes(kv_bar,2*TILE_BYTES);
    }
  };
  issue_kv();
  // Group g owns chunks g, g+2, ...; a one-chunk task leaves group 1 idle.
  const bool active=g<chunk_iterations;
  if(active) {gather(g);finish_gather();}
  barrier_wait(kv_bar);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  if(active)
  for(int local=0,chunk_iteration=g;chunk_iteration<chunk_iterations;++local,chunk_iteration+=GROUPS) {
    int* row_query=query_map[g][(chunk_iteration>>1)&1];
    int* row_rank=rank_map[g][(chunk_iteration>>1)&1];
    const int query=row_query[gtid],rank=row_rank[gtid];
    const bool has_next=chunk_iteration+GROUPS<chunk_iterations;
    if(local==0) issue_qk();
    wait_epoch(qk_bar,local&1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    // QK completion releases this group's single Q buffer for the next gather.
    if(has_next) gather(chunk_iteration+GROUPS);
  const int trow = (gwarp * 32) << 16;
  const float scale = 1.0f / sqrtf(float(D));
  float row_max = -FLT_MAX;
  float denom = 0.0f;
  {
    float cached[N];
    constexpr int WIDTH=32;
    #pragma unroll
    for(int col=0;col<N;col+=WIDTH) {
      float scores[WIDTH];
      load32(ts+trow+col,scores);
      #pragma unroll
      for(int j=0;j<WIDTH;++j) {
        cached[col+j]=scores[j];
        row_max=fmaxf(row_max,scores[j]);
      }
    }
    row_max*=scale;
    #pragma unroll
    for(int col=0;col<N;col+=16) {
      uint32_t packed[8];
      #pragma unroll
      for(int j=0;j<8;++j) {
        float p0,p1;
          p0=__expf(__fmaf_rn(cached[col+2*j],scale,-row_max));
          p1=__expf(__fmaf_rn(cached[col+2*j+1],scale,-row_max));
        denom+=p0+p1;
        packed[j]=pack2(p0,p1);
      }
      store8(ts+trow+col/2,packed);
    }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  group_sync(g);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");


  if (group_issuer(tid,g)) {
    const uint64_t v = mnmajor(v_addr);
    for (int micro = 0; micro < N / 16; ++micro)
      mma_ts(to, ts + micro * 8, v + micro * (D * 32 / 16),
             desc | (1U << 16), micro != 0);
    mma_commit(pv_bar);
  }
  wait_epoch(pv_bar,local&1);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");


    if(has_next) {
      finish_gather();
      // PV completion releases S/P; next QK writes S, not current O.
      issue_qk();
    }
  const float inv_denom = __frcp_rn(denom);
  const int head = kh * group_size + (gtid & group_mask);
  const int64_t slot = ((int64_t(b) * hq + head) * sq + (query >= 0 ? query : 0))
                     * TOPK + (rank >= 0 ? rank : 0);
  {
    // One thread owns one row: evacuate, stage, fence, bulk-copy, commit;
    // FP32 rows go in two 256-byte halves through the same staging row.
    typename S::T* row_stage = stage + gtid*S::ROW_ELEMS;
    #pragma unroll
    for(int pass=0;pass<S::PASSES;++pass) {
      bulk_wait_read();  // the previous copy has finished reading this staging row
      for(int col=0;col<S::HALF_ELEMS;col+=32) {
        float values[32];
        load32(to+trow+pass*S::HALF_ELEMS+col,values);
        if(query>=0) {
          if constexpr(BF16P) {
            #pragma unroll
            for(int j=0;j<32;j+=8) {
              const uint4 value={pack2(values[j]*inv_denom,values[j+1]*inv_denom),
                                 pack2(values[j+2]*inv_denom,values[j+3]*inv_denom),
                                 pack2(values[j+4]*inv_denom,values[j+5]*inv_denom),
                                 pack2(values[j+6]*inv_denom,values[j+7]*inv_denom)};
              *reinterpret_cast<uint4*>(row_stage+col+j)=value;
            }
          } else {
            #pragma unroll
            for(int j=0;j<32;j+=4) {
              const float4 value={values[j]*inv_denom,values[j+1]*inv_denom,
                                  values[j+2]*inv_denom,values[j+3]*inv_denom};
              *reinterpret_cast<float4*>(row_stage+col+j)=value;
            }
          }
        }
      }
      if (query >= 0) {
        if(pass==0) partial_lse[slot] = row_max + logf(denom);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        bulk_store(partial_o + slot * D + pass*S::HALF_ELEMS, smem_address(row_stage), S::COPY_BYTES);
      }
      bulk_commit();
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  group_sync(g);   // O accumulator evacuated before this group's next PV
  }
  if(active) bulk_wait_all();
  __syncthreads();   // both groups done with TMEM and shared memory
  if(warp==0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;"
                           :: "r"(tmem_base) : "memory");
}
} // namespace kv_split9_ops

namespace {
CUtensorMap v_folded_map(const nv_bfloat16* ptr,int height) {
  alignas(64) CUtensorMap map{};
  uint64_t dims[]={64,8,2,uint64_t(height/8)};
  uint64_t strides[]={256,128,2048};
  uint32_t box[]={64,8,2,16}, steps[]={1,1,1,1};
  kv0::cu_check(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,
      const_cast<nv_bfloat16*>(ptr),dims,strides,box,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

template<int MAX_CHUNKS,bool BF16P>
void launch9(const nv_bfloat16* q,const nv_bfloat16* k,const nv_bfloat16* v,
    const int32_t* tasks,const int32_t* qids,const int32_t* ranks,void* partial,float* lses,
    int b,int hq,int hkv,int sq,int skv,int ntasks,int nedges,cudaStream_t stream) {
  using S=kv_split9_ops::Staging<BF16P>;
  auto km=kv0::make_tmap(k,b*hkv*skv,128);auto vm=v_folded_map(v,b*hkv*skv);
  constexpr int bytes=kv_split9_ops::STAGE_OFFSET+kv_split9_ops::GROUPS*S::BYTES;   // 196 KiB
  auto fn=kv_split9_ops::main_kernel<MAX_CHUNKS,BF16P>;
  kv0::cuda_check(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  fn<<<ntasks,256,bytes,stream>>>(q,km,vm,tasks,qids,ranks,
                                reinterpret_cast<typename S::T*>(partial),lses,
                                b,hq,hkv,sq,skv,hq/hkv,nedges);
  kv0::cuda_check(cudaGetLastError());
}
}
// partial is float* for variant 0 and nv_bfloat16* for 1/2; the wrapper checks dtypes.
void kv9_launch(const nv_bfloat16* q,const nv_bfloat16* k,const nv_bfloat16* v,
    const int32_t* tasks,const int32_t* qids,const int32_t* ranks,void* partial,float* lses,
    int b,int hq,int hkv,int sq,int skv,int ntasks,int nedges,int variant,cudaStream_t stream) {
#define RUN(C,P) launch9<C,P>(q,k,v,tasks,qids,ranks,partial,lses,b,hq,hkv,sq,skv,ntasks,nedges,stream)
  switch(variant) {
    case 0: RUN(8,false);break;   // split8: exact FP32 partials
    case 1: RUN(8,true);break;    // split8_bf16
    case 2: RUN(16,true);break;   // split16_bf16: 8 chunks per group per task
    default: TORCH_CHECK(false,"round9 variant");
  }
#undef RUN
}
