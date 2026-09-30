// Round11: token-major layouts, TMA gather and TMA scatter, on round9's split.
// Round10's profile showed the per-row gather (2048 cp.async with 64-bit
// addressing) and the per-row epilogue (128 bulk copies) cost as many
// instructions as the whole softmax. MSA pays 1/16 of that because its Q and
// partial layouts keep a token's G heads contiguous. This kernel adopts that:
//   Q          [B, Sq, Hq, D]        -> a chunk's rows are 128/G contiguous
//                                        G-row blocks: 2 TMA loads per token
//   partial O  [B, Sq, 16, Hq, D]     -> 2 (BF16) or 4 (FP32) TMA stores per
//                                        token from a 128B-swizzled staging tile
// K/V, LSE [B,Hq,Sq,16] and final O [B,Hq,Sq,D] are unchanged. TMA boxes of G
// rows keep the 128B swizzle consistent only when G*128 B is a multiple of the
// 1 KiB swizzle atom, so TMA is used for G>=8; G<8 keeps round9's cp.async
// gather and per-row bulk epilogue with the new partial index. Arithmetic,
// rounding and Combine order are round9's: bit-identical to split8/split8_bf16.
#include "../primitives.cuh"
#include <math_constants.h>
#include <cfloat>
#include <cmath>
#include <type_traits>

namespace kv_tma11_ops {
using namespace kv0;
constexpr int GROUPS=2,GROUP_THREADS=128,GROUP_WARPS=4;
constexpr int PANEL_BYTES=M*128;          // 128 rows x 128 B: one swizzle panel of a tile
__device__ __forceinline__ void wait_epoch(int address,int phase) {
  asm volatile("{ .reg .pred p; WAIT: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; "
               "@!p bra WAIT; }" :: "r"(address),"r"(phase) : "memory");
}
__device__ __forceinline__ bool group_issuer(int tid,int g) {
  if ((tid >> 5) != GROUP_WARPS*g) return false;
  unsigned leader;
  asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.b32 %0, 1, 0, p; }"
               : "=r"(leader) :: "memory");
  return leader != 0;
}
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
// TMA store of one box from swizzled shared memory to the partial tensor.
__device__ __forceinline__ void tma_store(const CUtensorMap* map,int x,int y,int src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
               :: "l"(map),"r"(x),"r"(y),"r"(src) : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
template<int PENDING>
__device__ __forceinline__ void bulk_wait_read_allow() {
  asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(PENDING) : "memory");
}
__device__ __forceinline__ void bulk_wait_all() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
// 128B-swizzled staging: panel p, row r, 16-byte chunk c -> byte offset.
__device__ __forceinline__ int swizzled(int panel,int row,int chunk) {
  return panel*PANEL_BYTES+row*128+((chunk^(row&7))<<4);
}

template<bool BF16P>
struct Staging {
  using T=std::conditional_t<BF16P,nv_bfloat16,float>;
  static constexpr int ELEM=int(sizeof(T));
  static constexpr int PANEL_ELEMS=128/ELEM;          // columns per 128-byte panel: 64 bf16 / 32 fp32
  static constexpr int PANELS=D/PANEL_ELEMS;          // 2 / 4
  // BF16: one pass of two panels (32 KiB); the previous chunk's stores have long
  // finished reading. FP32: four single-panel passes through two alternating
  // 16 KiB halves, so a pass only waits for the stores issued two passes earlier.
  static constexpr int PASSES=BF16P?1:4;
  static constexpr int PANELS_PER_PASS=PANELS/PASSES; // 2 / 1
  static constexpr int COLS_PER_PASS=D/PASSES;        // 128 / 32
  static constexpr int PASS_STRIDE_BYTES=BF16P?0:PANEL_BYTES;   // half selected by pass&1
  static constexpr int PENDING_ALLOWED=BF16P?0:1;
  // Fallback (G<8) staging: 272-byte rows, half rows for FP32 (round9).
  static constexpr int ROW_ELEMS=(BF16P?D:D/2)+(BF16P?8:4);
  static constexpr int ROW_BYTES=ROW_ELEMS*ELEM;
  static constexpr int COPY_BYTES=(BF16P?D:D/2)*ELEM;
  static constexpr int BYTES=M*ROW_BYTES;             // 34 KiB >= 2 panels (32 KiB)
  static_assert(ROW_BYTES==272 && COPY_BYTES==256 && 2*PANEL_BYTES<=BYTES,"staging geometry");
};
constexpr int Q_OFFSET=0,K_OFFSET=2*TILE_BYTES,V_OFFSET=3*TILE_BYTES,STAGE_OFFSET=4*TILE_BYTES;

template<int MAX_CHUNKS,bool BF16P,bool TMA>
__global__ __launch_bounds__(256) void main_kernel(
    const nv_bfloat16* __restrict__ q,
    const __grid_constant__ CUtensorMap q_map,
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap v_map,
    const __grid_constant__ CUtensorMap p_map,
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
  const int rows_per_chunk=M>>group_log;   // tokens per chunk
  const int chunk_iterations=(count+rows_per_chunk-1)/rows_per_chunk;
  extern __shared__ __align__(1024) unsigned char storage[];
  const int q_addr=smem_address(storage)+Q_OFFSET+g*TILE_BYTES;
  const int k_addr=smem_address(storage)+K_OFFSET,v_addr=smem_address(storage)+V_OFFSET;
  unsigned char* stage_ptr=storage+STAGE_OFFSET+g*S::BYTES;
  const int stage=smem_address(stage_ptr);
  __shared__ int query_map[GROUPS][2][M],rank_map[GROUPS][2][M];
  __shared__ uint64_t kv_ready,qk_done[GROUPS],pv_done[GROUPS],q_ready[GROUPS];
  __shared__ int tmem_base;
  const int kv_bar=smem_address(&kv_ready);
  const int qk_bar=smem_address(&qk_done[g]),pv_bar=smem_address(&pv_done[g]),q_bar=smem_address(&q_ready[g]);
  if(tid==0) {
    barrier_init(kv_bar);
    for(int i=0;i<GROUPS;++i) {barrier_init(smem_address(&qk_done[i]));barrier_init(smem_address(&pv_done[i]));barrier_init(smem_address(&q_ready[i]));}
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
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
  // Fill this chunk's query/rank maps (one row per thread), then load Q.
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
    if constexpr(TMA) {
      // One G-row box per (token, panel); padding tokens are skipped and the
      // expected byte count shrinks accordingly. Q rows of a token's G heads
      // are contiguous in [B,Sq,Hq,D], and the box lands swizzled.
      if(group_issuer(tid,g)) {
        int bytes=0;
        for(int t=0;t<rows_per_chunk;++t) {
          const int qi=query_map[g][bank][t<<group_log];
          if(qi>=0) {
            const int y=(b*sq+qi)*hq+kh*group_size;
            for(int panel=0;panel<2;++panel)
              tma_copy(q_addr+panel*PANEL_BYTES+(t<<group_log)*128,&q_map,panel*64,y,q_bar);
            bytes+=group_size*256;
          }
        }
        expect_bytes(q_bar,bytes);
      }
    } else {
      for(int vector=gtid;vector<M*D/8;vector+=GROUP_THREADS) {
        const int row=vector/16,col=(vector%16)*8;
        const int qi=query_map[g][bank][row],head=kh*group_size+(row&group_mask);
        const int offset=(col/64)*M*64+row*64+((col%64)^((row%8)*8));
        const auto* src=q+((int64_t(b)*sq+(qi>=0?qi:0))*hq+head)*D+col;
        const int ignore=qi<0;
        asm volatile("{ .reg .pred zero; setp.ne.b32 zero, %2, 0; "
                     "cp.async.cg.shared.global [%0], [%1], 16, zero; }"
                     :: "r"(q_addr+offset*2),"l"(src),"r"(ignore) : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  };
  auto finish_gather = [&](int local) {
    if constexpr(TMA) {
      wait_epoch(q_bar,local&1);
    } else {
      asm volatile("cp.async.wait_group 0; fence.proxy.async.shared::cta;" ::: "memory");
      group_sync(g);
    }
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
  const bool active=g<chunk_iterations;
  if(active) {gather(g);finish_gather(0);}
  barrier_wait(kv_bar);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  if(active)
  for(int local=0,chunk_iteration=g;chunk_iteration<chunk_iterations;++local,chunk_iteration+=GROUPS) {
    const int bank=(chunk_iteration>>1)&1;
    int* row_query=query_map[g][bank];
    int* row_rank=rank_map[g][bank];
    const int query=row_query[gtid],rank=row_rank[gtid];
    const bool has_next=chunk_iteration+GROUPS<chunk_iterations;
    if(local==0) issue_qk();
    wait_epoch(qk_bar,local&1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
      finish_gather(local+1);
      issue_qk();
    }
  const float inv_denom = __frcp_rn(denom);
  const int head = kh * group_size + (gtid & group_mask);
  // LSE keeps the [B,Hq,Sq,16] layout; partial O is token-major [B,Sq,16,Hq,D].
  const int64_t lse_slot = ((int64_t(b) * hq + head) * sq + (query >= 0 ? query : 0))
                         * TOPK + (rank >= 0 ? rank : 0);
  if constexpr(TMA) {
    #pragma unroll
    for(int pass=0;pass<S::PASSES;++pass) {
      // The stores that last used this staging half must have finished reading it.
      const int half=(pass&1)*S::PASS_STRIDE_BYTES;
      if(group_issuer(tid,g)) bulk_wait_read_allow<S::PENDING_ALLOWED>();
      group_sync(g);
      #pragma unroll
      for(int col=0;col<S::COLS_PER_PASS;col+=32) {
        float values[32];
        load32(to+trow+pass*S::COLS_PER_PASS+col,values);
        if(query>=0) {
          const int panel=col/S::PANEL_ELEMS;                 // panel within this pass
          if constexpr(BF16P) {
            #pragma unroll
            for(int j=0;j<32;j+=8) {
              const int chunk=((col%S::PANEL_ELEMS)+j)/8;
              const uint4 value={pack2(values[j]*inv_denom,values[j+1]*inv_denom),
                                 pack2(values[j+2]*inv_denom,values[j+3]*inv_denom),
                                 pack2(values[j+4]*inv_denom,values[j+5]*inv_denom),
                                 pack2(values[j+6]*inv_denom,values[j+7]*inv_denom)};
              *reinterpret_cast<uint4*>(stage_ptr+half+swizzled(panel,gtid,chunk))=value;
            }
          } else {
            #pragma unroll
            for(int j=0;j<32;j+=4) {
              const int chunk=((col%S::PANEL_ELEMS)+j)/4;
              const float4 value={values[j]*inv_denom,values[j+1]*inv_denom,
                                  values[j+2]*inv_denom,values[j+3]*inv_denom};
              *reinterpret_cast<float4*>(stage_ptr+half+swizzled(panel,gtid,chunk))=value;
            }
          }
        }
      }
      if(pass==0 && query>=0) partial_lse[lse_slot] = row_max + logf(denom);
      // Generic staging writes -> async proxy, then one thread scatters G-row boxes.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      group_sync(g);
      if(group_issuer(tid,g)) {
        for(int t=0;t<rows_per_chunk;++t) {
          const int qi=row_query[t<<group_log],r=row_rank[t<<group_log];
          if(qi>=0) {
            const int y=((b*sq+qi)*TOPK+r)*hq+kh*group_size;
            for(int panel=0;panel<S::PANELS_PER_PASS;++panel)
              tma_store(&p_map,(pass*S::PANELS_PER_PASS+panel)*S::PANEL_ELEMS,y,
                        stage+half+panel*PANEL_BYTES+(t<<group_log)*128);
          }
        }
        bulk_commit();
      }
    }
  } else {
    // Round9 per-row epilogue with the token-major partial index.
    typename S::T* row_stage=reinterpret_cast<typename S::T*>(stage_ptr)+gtid*S::ROW_ELEMS;
    const int64_t o_slot=((int64_t(b)*sq+(query>=0?query:0))*TOPK+(rank>=0?rank:0))*hq+head;
    constexpr int ROW_PASSES=BF16P?1:2,ROW_COLS=D/ROW_PASSES;   // round9: 256-byte half rows
    #pragma unroll
    for(int pass=0;pass<ROW_PASSES;++pass) {
      bulk_wait_read();
      for(int col=0;col<ROW_COLS;col+=32) {
        float values[32];
        load32(to+trow+pass*ROW_COLS+col,values);
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
        if(pass==0) partial_lse[lse_slot] = row_max + logf(denom);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        bulk_store(partial_o + o_slot * D + pass*ROW_COLS, smem_address(row_stage), S::COPY_BYTES);
      }
      bulk_commit();
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  group_sync(g);
  }
  if(active) {
    if constexpr(TMA) { if(group_issuer(tid,g)) bulk_wait_all(); }
    else bulk_wait_all();
  }
  __syncthreads();
  if(warp==0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;"
                           :: "r"(tmem_base) : "memory");
}
} // namespace kv_tma11_ops

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
// 2D map over a [rows, D] tensor with 128-byte-wide boxes of `rows_per_box` rows.
CUtensorMap row_map(const void* ptr,int64_t rows,int rows_per_box,CUtensorMapDataType type,int elem_bytes) {
  alignas(64) CUtensorMap map{};
  uint64_t dims[]={uint64_t(kv0::D),uint64_t(rows)};
  uint64_t strides[]={uint64_t(kv0::D*elem_bytes)};
  uint32_t box[]={uint32_t(128/elem_bytes),uint32_t(rows_per_box)}, steps[]={1,1};
  kv0::cu_check(cuTensorMapEncodeTiled(&map,type,2,const_cast<void*>(ptr),dims,strides,box,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

template<int MAX_CHUNKS,bool BF16P,bool TMA>
void launch11(const nv_bfloat16* q,const nv_bfloat16* k,const nv_bfloat16* v,
    const int32_t* tasks,const int32_t* qids,const int32_t* ranks,void* partial,float* lses,
    int b,int hq,int hkv,int sq,int skv,int ntasks,int nedges,cudaStream_t stream) {
  using S=kv_tma11_ops::Staging<BF16P>;
  const int g=hq/hkv;
  auto km=kv0::make_tmap(k,b*hkv*skv,128);auto vm=v_folded_map(v,b*hkv*skv);
  // Maps are encoded for every launch; with G<8 the kernel never dereferences them.
  const int box_rows=TMA?g:8;
  auto qm=row_map(q,int64_t(b)*sq*hq,box_rows,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2);
  auto pm=row_map(partial,int64_t(b)*sq*kv0::TOPK*hq,box_rows,
                  BF16P?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_FLOAT32,S::ELEM);
  constexpr int bytes=kv_tma11_ops::STAGE_OFFSET+kv_tma11_ops::GROUPS*S::BYTES;   // 196 KiB
  auto fn=kv_tma11_ops::main_kernel<MAX_CHUNKS,BF16P,TMA>;
  kv0::cuda_check(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  fn<<<ntasks,256,bytes,stream>>>(q,qm,km,vm,pm,tasks,qids,ranks,
                                reinterpret_cast<typename S::T*>(partial),lses,
                                b,hq,hkv,sq,skv,g,nedges);
  kv0::cuda_check(cudaGetLastError());
}
}
// Q is [B,Sq,Hq,D]; partial is [B,Sq,16,Hq,D] float (variant 0) or bf16 (1, 2).
void kv11_launch(const nv_bfloat16* q,const nv_bfloat16* k,const nv_bfloat16* v,
    const int32_t* tasks,const int32_t* qids,const int32_t* ranks,void* partial,float* lses,
    int b,int hq,int hkv,int sq,int skv,int ntasks,int nedges,int variant,cudaStream_t stream) {
  const bool tma=(hq/hkv)>=8;
#define RUN(C,P) if(tma) launch11<C,P,true>(q,k,v,tasks,qids,ranks,partial,lses,b,hq,hkv,sq,skv,ntasks,nedges,stream); \
                 else launch11<C,P,false>(q,k,v,tasks,qids,ranks,partial,lses,b,hq,hkv,sq,skv,ntasks,nedges,stream)
  switch(variant) {
    case 0: RUN(8,false);break;   // tma8: exact FP32 partials
    case 1: RUN(8,true);break;    // tma8_bf16
    case 2: RUN(16,true);break;   // tma16_bf16
    default: TORCH_CHECK(false,"round11 variant");
  }
#undef RUN
}
