// Kernel 14 focus: folded persistent-core checkpoint.
// Adds persistent CTA scheduling, role-local persistent loops, and
// __launch_bounds__(TB_SIZE, 1), while preserving the direct final-output path
// from the pre-persistent kernel.
// It preserves the conservative lockstep boundary: the next work item does not start
// until load, MMA, softmax, and correction finish the previous work item.

#include "common.cuh"
#include "approximations.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// tcgen05 + TMA + TMEM attention with q_stage=2 + early-PV.
//
// Assumptions:
//   - head_dim == 128
//   - BLOCK_M == 128 queries per Q stage
//   - BLOCK_N == 128 keys per KV tile
//   - q_stage == 2, so each CTA covers 256 query rows
//   - len_q % 256 == 0
//   - len_kv % 128 == 0
//   - dense, non-causal, no masking
//   - inputs/outputs are bf16
//
// Steady-state dataflow:
//   1) load warp loads Q0/Q1 once and drives one unified KV[3] token stream
//      in forward tile order: K(0), V(0), K(1), V(1), ...
//   2) MMA warp computes S0 = Q0 @ K^T, then S1 = Q1 @ K^T using the same K tile
//   3) softmax warps for stage0/stage1 independently consume S0/S1 and write packed P0/P1
//      as a 96-column prefix followed by a 32-column tail
//   4) correction warps rescale O0/O1 using the per-row rescale factors from softmax
//   5) MMA warp waits for joined Pq-prefix + Oq-safe, starts prefix PV,
//      then waits for full Pq and finishes tail PV
//   6) after the KV loop: softmax reads O from TMEM, normalizes by rowsum,
//      and writes directly to global memory
//
// MMA steady-state schedule:
//   - seed tile 0 by computing QK0(0), QK1(0)
//   - then for each KV tile i:
//       * consume PV0(i), then immediately produce QK0(i+1)
//       * consume PV1(i), then immediately produce QK1(i+1)
//
// Important tradeoff relative to the one-Q compute-pipeline sidebranch:
//   - The sidebranch used spare TMEM S/P capacity to overlap QK(i+1) with row work for tile i
//     for a single Q tile.
//   - Kernel 5 repurposes that TMEM space for q_stage=2: S0/S1 + O0/O1 fill the 512-column TMEM budget.
//   - So q_stage=2 gains K/V reuse across two Q tiles, but it does not keep the old same-Q
//     time-axis compute lookahead from kernel 6.

namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int HEAD_DIM = 128;
constexpr int MMA_K = 16;

namespace q_stage {
constexpr int STAGES = 2;
}  // namespace q_stage
constexpr int P_CHUNKS = BLOCK_N / MMA_K;          // 8 when BLOCK_N=128
constexpr int P_HEAD_CHUNKS = (3 * P_CHUNKS) / 4;  // 6 when BLOCK_N=128
constexpr int P_SPLIT_N16 = P_CHUNKS - P_HEAD_CHUNKS;  // 2 when BLOCK_N=128
constexpr int P_TAIL_N8 = P_SPLIT_N16 * 2;  // 4 when BLOCK_N=128
constexpr int P_HEAD_N8 = BLOCK_N / 8 - P_TAIL_N8;  // 12 when BLOCK_N=128
constexpr int E2E_FREQ = 16;
constexpr int E2E_RES = 5;
constexpr int E2E_FRG_TILE = 32;
constexpr int E2E_FRG_LIMIT = 0;
constexpr float RESCALE_THRESHOLD_LOG2 = 8.0f;
static_assert(P_HEAD_CHUNKS + P_SPLIT_N16 == P_CHUNKS, "split must partition P chunks");
static_assert(P_HEAD_N8 + P_TAIL_N8 == BLOCK_N / 8, "P head/tail must cover the score tile");
static_assert(BLOCK_N % E2E_FRG_TILE == 0, "BLOCK_N must divide the E2E fragment tile");
static_assert(P_SPLIT_N16 == 2 && P_CHUNKS == 8,
              "high-prefix loop-PV carrier assumes the current 96/32 split");

constexpr int SOFTMAX_WARPS_PER_STAGE = 4;
constexpr int SOFTMAX_WARP_COUNT = q_stage::STAGES * SOFTMAX_WARPS_PER_STAGE;
constexpr int CORR_WARP_BASE = SOFTMAX_WARP_COUNT;
constexpr int CORR_WARP_COUNT = 4;
constexpr int LOAD_WARP_ID = CORR_WARP_BASE + CORR_WARP_COUNT;
constexpr int MMA_WARP_ID = LOAD_WARP_ID + 1;
constexpr int NUM_WARPS = MMA_WARP_ID + 1;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int PERSISTENT_CTAS_PER_SM = 1;

constexpr int BF16_BYTES = int(sizeof(nv_bfloat16));
constexpr int MMA_K_BYTES = MMA_K * BF16_BYTES;         // 32
// In this kernel's matrix-coordinate view, SW128 operand panels are 8 rows x 64 bf16 cols.
// K-major and MN-major descriptors organize/interpret those panels differently.
constexpr int SW128_BYTES = 128;
constexpr int SW128_ATOM_ROWS = 8;
constexpr int SW128_ATOM_COLS = SW128_BYTES / BF16_BYTES;  // 64 bf16
#include "smem_layout_swizzle128.cuh"
static_assert(HEAD_DIM % SW128_ATOM_COLS == 0, "HEAD_DIM must be divisible by SW128 width");
static_assert(MMA_K_BYTES == 32, "MMA_K_BYTES must match tcgen05 microstep bytes");

constexpr int Q_TILE_BYTES = BLOCK_M * HEAD_DIM * BF16_BYTES;
constexpr int KV_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;

constexpr int TMEM_S_COLS = BLOCK_N;
constexpr int TMEM_O_COLS = HEAD_DIM;
constexpr int TMEM_ALLOC_COLS = q_stage::STAGES * TMEM_S_COLS + q_stage::STAGES * TMEM_O_COLS;
// Full 512-column TMEM budget is consumed by S0/S1 + O0/O1.
// That is why this kernel cannot also keep the sidebranch extra S/P time-stage.
static_assert(TMEM_ALLOC_COLS == 512, "q_stage=2 should pack S0/S1 + O0/O1 into 512 TMEM cols");

namespace q_stage {

__device__ __forceinline__ int row0(int q_row0_base, int q) {
  return q_row0_base + q * BLOCK_M;
}

__device__ __forceinline__ int q_smem(int Q_smem, int q) {
  return Q_smem + q * Q_TILE_BYTES;
}

__device__ __forceinline__ int tmem_s(int taddr_s_base, int q) {
  return taddr_s_base + q * TMEM_S_COLS;
}

__device__ __forceinline__ int tmem_o(int taddr_o_base, int q) {
  return taddr_o_base + q * TMEM_O_COLS;
}

__device__ __forceinline__ int mbar_addr(int base, int q) {
  return base + q * int(sizeof(uint64_t));
}

}  // namespace q_stage

namespace kv_ring {

constexpr int STAGES = 3;
constexpr int TOKENS_PER_KV_TILE = 2;
constexpr int K_KIND = 0;
constexpr int V_KIND = 1;

__device__ __forceinline__ int logical_token(int kv_tile, int kind) {
  return TOKENS_PER_KV_TILE * kv_tile + kind;
}

__device__ __forceinline__ int physical_stage(int token) {
  return token % STAGES;
}

__device__ __forceinline__ int token_tile(int token) {
  return token / TOKENS_PER_KV_TILE;
}

__device__ __forceinline__ int stage_smem_addr(int KV_smem, int stage) {
  return KV_smem + stage * KV_TILE_BYTES;
}

__device__ __forceinline__ int stage_mbar_addr(int mbar_kv_ready_base, int stage) {
  return mbar_kv_ready_base + stage * int(sizeof(uint64_t));
}

__device__ __forceinline__ bool is_k_token(int token) {
  return (token % TOKENS_PER_KV_TILE) == K_KIND;
}

__device__ __forceinline__ void load_token(int token, int batch_id, int len_kv, int KV_smem,
                                           int mbar_kv_ready_base, const CUtensorMap* K_tmap,
                                           const CUtensorMap* V_tmap) {
  const int stage = physical_stage(token);
  const int tile = token_tile(token);
  const int kv_row0 = batch_id * len_kv + tile * BLOCK_N;
  const int mbar_kv_ready_addr = stage_mbar_addr(mbar_kv_ready_base, stage);
  const int smem_addr = stage_smem_addr(KV_smem, stage);

  if (is_k_token(token)) {
    swizzle128::tma_load_kmajor(smem_addr, K_tmap, BLOCK_N, kv_row0, mbar_kv_ready_addr);
  } else {
    swizzle128::tma_load_mnmajor(smem_addr, V_tmap, kv_row0, mbar_kv_ready_addr);
  }
  mbarrier::arrive_expect_tx(mbar_kv_ready_addr, KV_TILE_BYTES);
}

}  // namespace kv_ring

__device__ __forceinline__ void issue_qk_mma(int taddr_s_stage, int Q_stage_smem,
                                             int K_stage_smem, uint32_t i_desc_qk) {
  // outer loop selects SW128 64-wide block
  constexpr uint64_t qk_desc_micro_step = MMA_K_BYTES / 16;
  constexpr uint64_t qk_a_desc_panel_step = (BLOCK_M * SW128_BYTES) / 16;
  constexpr uint64_t qk_b_desc_panel_step = (BLOCK_N * SW128_BYTES) / 16;
  uint64_t a_desc_panel = swizzle128::desc_kmajor(Q_stage_smem);
  uint64_t b_desc_panel = swizzle128::desc_kmajor(K_stage_smem);
  for (int k1 = 0; k1 < HEAD_DIM / SW128_ATOM_COLS; ++k1) {
    uint64_t a_desc = a_desc_panel;
    uint64_t b_desc = b_desc_panel;
    // inner loop selects MMA_K=16 chunk inside that block
    for (int k2 = 0; k2 < SW128_ATOM_COLS / MMA_K; ++k2) {
      const int enable_input_d = (k1 == 0 && k2 == 0) ? 0 : 1;
      tcgen05::mma_f16(taddr_s_stage, a_desc, b_desc, i_desc_qk, enable_input_d);
      a_desc += qk_desc_micro_step;
      b_desc += qk_desc_micro_step;
    }
    a_desc_panel += qk_a_desc_panel_step;
    b_desc_panel += qk_b_desc_panel_step;
  }
}

}  // namespace

__global__ __launch_bounds__(TB_SIZE, 1) void attention_tcgen05_v14_kernel(
    const __grid_constant__ CUtensorMap Q_tmap, const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap, nv_bfloat16* __restrict__ O_ptr, int len_q,
    int len_kv, int total_work_tiles) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const bool is_softmax_warp = (warp_id < SOFTMAX_WARP_COUNT);
  const bool is_corr_warp = (warp_id >= CORR_WARP_BASE && warp_id < CORR_WARP_BASE + CORR_WARP_COUNT);
  const bool is_load_warp = (warp_id == LOAD_WARP_ID);
  const bool is_mma_warp = (warp_id == MMA_WARP_ID);

  extern __shared__ __align__(1024) char smem_raw[];

  const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
  const int Q_smem = smem_base;
  const int KV_smem = Q_smem + q_stage::STAGES * Q_TILE_BYTES;

  // Stage barriers:
  //   q-ready:            load warp -> mma warp (Q0/Q1 loaded once)
  //   kv-ready[s]:        load warp -> mma warp for the unified K/V token stream
  //   qk-done[q]:         mma warp -> softmax warps for Q stage q
  //   stats-ready[q]:     softmax warps -> correction warps for Q stage q
  //   p-half-ready[q]:    joined softmax/correction handoff once prefix P is stored and Oq is safe
  //   p-ready[q]:         softmax warps -> mma warp for Q stage q
  //   pv-done[q]:         mma warp -> correction/load warp for Q stage q
  //   setup-done:         load warp -> compute warps after per-work barrier setup
  //   work-done:          all warp roles -> load/MMA so next setup/dealloc is lockstep-safe
  __shared__ uint64_t mbar_q_ready[1];
  __shared__ uint64_t mbar_setup_done[1];
  __shared__ uint64_t mbar_work_done[1];
  __shared__ uint64_t mbar_qk_done[q_stage::STAGES];
  __shared__ uint64_t mbar_stats_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_p_half_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_p_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_pv_done[q_stage::STAGES];
  __shared__ uint64_t mbar_kv_ready[kv_ring::STAGES];
  __shared__ float rescale_smem[q_stage::STAGES][BLOCK_M];
  __shared__ int tmem_addr[1];

  const int mbar_q_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_q_ready));
  const int mbar_setup_done_addr = static_cast<int>(__cvta_generic_to_shared(mbar_setup_done));
  const int mbar_work_done_addr =
      static_cast<int>(__cvta_generic_to_shared(mbar_work_done));
  const int mbar_qk_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_qk_done));
  const int mbar_stats_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_stats_ready));
  const int mbar_p_half_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_p_half_ready));
  const int mbar_p_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_p_ready));
  const int mbar_pv_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_pv_done));
  const int mbar_kv_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_kv_ready));

  const int q_tile_pairs_per_batch = len_q / (q_stage::STAGES * BLOCK_M);


  if (is_load_warp && elect_sync()) {
    mbarrier::init(mbar_setup_done_addr, 1);
    mbarrier::init(mbar_work_done_addr, NUM_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (is_mma_warp) {
    // TMEM layout:
    //   S0 at [0..127], S1 at [128..255], O0 at [256..383], O1 at [384..511]
    //   Pq overlays upper half of Sq at +64 columns.
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :
                 : "r"(tmem_smem_addr), "r"(TMEM_ALLOC_COLS)
                 : "memory");
  }
  __syncthreads();

  const int taddr_base = tmem_addr[0];
  const int taddr_s_base = taddr_base;
  const int taddr_o_base = taddr_s_base + q_stage::STAGES * TMEM_S_COLS;

  constexpr uint32_t i_desc_qk =
      (1U << 4) | (1U << 7) | (1U << 10) | ((BLOCK_N >> 3) << 17) | ((BLOCK_M >> 4) << 24);
  constexpr uint32_t i_desc_pv = i_desc_qk | (1U << 16);

  const int kv_tiles = len_kv / BLOCK_N;
  const int work_count = (total_work_tiles - 1 - blockIdx.x) / gridDim.x + 1;

  // Role-local persistent loops keep kernel 14's lockstep work-item boundary, but avoid
  // presenting ptxas with one giant loop body containing every warp role.
  if (is_load_warp && elect_sync()) {
    int phase_work_done = 0;
    for (int bid = blockIdx.x, work_iter = 0; bid < total_work_tiles;
         bid += gridDim.x, ++work_iter) {
      const int batch_id = bid / q_tile_pairs_per_batch;
      const int q_tile_pair_id = bid - batch_id * q_tile_pairs_per_batch;
      const int q_row0_base = batch_id * len_q + q_tile_pair_id * (q_stage::STAGES * BLOCK_M);

      if (work_iter > 0) {
        mbarrier::wait(mbar_work_done_addr, phase_work_done);
        phase_work_done ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
      }
      mbarrier::init(mbar_q_ready_addr, 1);
      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
        const int mbar_stats_ready_addr = q_stage::mbar_addr(mbar_stats_ready_base, q);
        const int mbar_p_half_ready_addr = q_stage::mbar_addr(mbar_p_half_ready_base, q);
        const int mbar_p_ready_addr = q_stage::mbar_addr(mbar_p_ready_base, q);
        const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
        mbarrier::init(mbar_qk_done_addr, 1);
        mbarrier::init(mbar_stats_ready_addr, SOFTMAX_WARPS_PER_STAGE);
        mbarrier::init(mbar_p_half_ready_addr, SOFTMAX_WARPS_PER_STAGE + CORR_WARP_COUNT);
        mbarrier::init(mbar_p_ready_addr, SOFTMAX_WARPS_PER_STAGE);
        mbarrier::init(mbar_pv_done_addr, 1);
      }
      for (int st = 0; st < kv_ring::STAGES; ++st) {
        const int mbar_kv_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, st);
        mbarrier::init(mbar_kv_ready_addr, 1);
      }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      mbarrier::arrive(mbar_setup_done_addr);

      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
        const int q_row0_stage = q_stage::row0(q_row0_base, q);
        swizzle128::tma_load_kmajor(Q_stage_smem, &Q_tmap, BLOCK_M, q_row0_stage,
                                          mbar_q_ready_addr);
      }
      constexpr int cp_size_q = q_stage::STAGES * Q_TILE_BYTES;
      mbarrier::arrive_expect_tx(mbar_q_ready_addr, cp_size_q);

      // Load warp recycles K/V only after the last q-stage consumer stream.
      int phase_k_recycle = 0;
      int phase_v_recycle = 0;
      const int kv_tokens = kv_ring::TOKENS_PER_KV_TILE * kv_tiles;
      const int prefill_tokens = (kv_tokens < kv_ring::STAGES) ? kv_tokens : kv_ring::STAGES;
      for (int token = 0; token < prefill_tokens; ++token) {
        kv_ring::load_token(token, batch_id, len_kv, KV_smem, mbar_kv_ready_base, &K_tmap, &V_tmap);
      }

      const int last_q = q_stage::STAGES - 1;
      const int mbar_last_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, last_q);
      const int mbar_last_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, last_q);
      for (int token = prefill_tokens; token < kv_tokens; ++token) {
        const int recycle_token = token - kv_ring::STAGES;
        if (kv_ring::is_k_token(recycle_token)) {
          mbarrier::wait(mbar_last_qk_done_addr, phase_k_recycle);
          phase_k_recycle ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        } else {
          mbarrier::wait(mbar_last_pv_done_addr, phase_v_recycle);
          phase_v_recycle ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        }

        kv_ring::load_token(token, batch_id, len_kv, KV_smem, mbar_kv_ready_base, &K_tmap, &V_tmap);
      }

      mbarrier::arrive(mbar_work_done_addr);
    }
  }

  // MMA warp: consume the unified K/V token stream and braid PV(q,i) -> QK(q,i+1).
  if (is_mma_warp) {
    for (int bid = blockIdx.x, work_iter = 0; bid < total_work_tiles;
         bid += gridDim.x, ++work_iter) {

      mbarrier::wait(mbar_setup_done_addr, work_iter & 1);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      if (elect_sync()) {
        mbarrier::wait(mbar_q_ready_addr, 0);
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
      }

      if (elect_sync()) {
        int phase_kv_ready[kv_ring::STAGES] = {0};
        int phase_p_half_ready[q_stage::STAGES] = {0, 0};
        int phase_p_ready[q_stage::STAGES] = {0, 0};
        const int k_token0 = kv_ring::logical_token(0, kv_ring::K_KIND);
        const int k_stage0 = kv_ring::physical_stage(k_token0);
        const int mbar_kv_ready_addr0 = kv_ring::stage_mbar_addr(mbar_kv_ready_base, k_stage0);
        const int K_stage_smem0 = kv_ring::stage_smem_addr(KV_smem, k_stage0);

        mbarrier::wait(mbar_kv_ready_addr0, phase_kv_ready[k_stage0]);
        phase_kv_ready[k_stage0] ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        for (int q = 0; q < q_stage::STAGES; ++q) {
          const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
          const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
          const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);

          issue_qk_mma(taddr_s_stage, Q_stage_smem, K_stage_smem0, i_desc_qk);

          tcgen05::commit_arrive(mbar_qk_done_addr);
        }

        for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
          const int v_token = kv_ring::logical_token(kv_tile, kv_ring::V_KIND);
          const int v_stage = kv_ring::physical_stage(v_token);
          const bool first_kv = (kv_tile == 0);
          const int mbar_kv_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, v_stage);
          const int V_stage_smem = kv_ring::stage_smem_addr(KV_smem, v_stage);

          const int next_tile = kv_tile + 1;
          const bool have_next = (next_tile < kv_tiles);
          int next_k_stage = 0;
          int next_mbar_kv_ready_addr = 0;
          int next_K_stage_smem = 0;
          if (have_next) {
            const int next_k_token = kv_ring::logical_token(next_tile, kv_ring::K_KIND);
            next_k_stage = kv_ring::physical_stage(next_k_token);
            next_mbar_kv_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, next_k_stage);
            next_K_stage_smem = kv_ring::stage_smem_addr(KV_smem, next_k_stage);
          }

          mbarrier::wait(mbar_kv_ready_addr, phase_kv_ready[v_stage]);
          phase_kv_ready[v_stage] ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

          for (int q = 0; q < q_stage::STAGES; ++q) {
            const int mbar_p_half_ready_addr = q_stage::mbar_addr(mbar_p_half_ready_base, q);
            const int mbar_p_ready_addr = q_stage::mbar_addr(mbar_p_ready_base, q);
            const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
            const int taddr_p_stage = q_stage::tmem_s(taddr_s_base, q) + BLOCK_N / 2;
            const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);


            mbarrier::wait(mbar_p_half_ready_addr, phase_p_half_ready[q]);
            phase_p_half_ready[q] ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            {
              const int enable_input_d_first = (first_kv ? 0 : 1);
              constexpr int pv_v_chunk_bytes = HEAD_DIM * MMA_K_BYTES;
              constexpr uint64_t pv_desc_chunk_step = pv_v_chunk_bytes / 16;
              uint64_t b_desc = swizzle128::desc_mnmajor(
                  V_stage_smem + P_SPLIT_N16 * pv_v_chunk_bytes, HEAD_DIM);
#pragma unroll
              for (int k = P_SPLIT_N16; k < P_CHUNKS; ++k) {
                const int taddr_a = taddr_p_stage + k * 8;
                tcgen05::mma_f16_a_tmem(taddr_o_stage, taddr_a, b_desc, i_desc_pv,
                                       (k == P_SPLIT_N16 ? enable_input_d_first : 1));
                b_desc += pv_desc_chunk_step;
              }

              mbarrier::wait(mbar_p_ready_addr, phase_p_ready[q]);
              phase_p_ready[q] ^= 1;
              asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

              b_desc = swizzle128::desc_mnmajor(V_stage_smem, HEAD_DIM);
#pragma unroll
              for (int k = 0; k < P_SPLIT_N16; ++k) {
                const int taddr_a = taddr_p_stage + k * 8;
                tcgen05::mma_f16_a_tmem(taddr_o_stage, taddr_a, b_desc, i_desc_pv, 1);
                b_desc += pv_desc_chunk_step;
              }
            }

            tcgen05::commit_arrive(mbar_pv_done_addr);

            if (have_next) {
              // The next K token is shared by q0 and q1, so consume its ready phase once.
              if (q == 0) {
                mbarrier::wait(next_mbar_kv_ready_addr, phase_kv_ready[next_k_stage]);
                phase_kv_ready[next_k_stage] ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
              }

              const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
              const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
              const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);

              issue_qk_mma(taddr_s_stage, Q_stage_smem, next_K_stage_smem, i_desc_qk);

              tcgen05::commit_arrive(mbar_qk_done_addr);
            }
          }
        }
      }

      __syncwarp();
      if (lane_id == 0) {
        mbarrier::arrive(mbar_work_done_addr);
      }
    }
  }

  // Softmax warps: one 4-warp group per Q stage.
  if (is_softmax_warp) {
    // Each Q stage still has 4 softmax warps covering rows 0..127
    // in slabs of 32 rows.
    const int softmax_stage = warp_id / SOFTMAX_WARPS_PER_STAGE;
    const int softmax_slab = warp_id % SOFTMAX_WARPS_PER_STAGE;
    const int softmax_row_tid = softmax_slab * WARP_SIZE + lane_id;

    for (int bid = blockIdx.x, work_iter = 0; bid < total_work_tiles;
         bid += gridDim.x, ++work_iter) {
      const int batch_id = bid / q_tile_pairs_per_batch;
      const int q_tile_pair_id = bid - batch_id * q_tile_pairs_per_batch;
      const int q_row0_base = batch_id * len_q + q_tile_pair_id * (q_stage::STAGES * BLOCK_M);

      mbarrier::wait(mbar_setup_done_addr, work_iter & 1);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      // This softmax warp belongs to one fixed q-stage, so it only needs one
      // local phase cursor for that qk_done stream.
      int phase_qk_done = 0;

      // q-stage-local resources for this softmax warp; invariant across all KV tiles.
      const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, softmax_stage);
      const int mbar_stats_ready_addr = q_stage::mbar_addr(mbar_stats_ready_base, softmax_stage);
      const int mbar_p_half_ready_addr = q_stage::mbar_addr(mbar_p_half_ready_base, softmax_stage);
      const int mbar_p_ready_addr = q_stage::mbar_addr(mbar_p_ready_base, softmax_stage);
      const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, softmax_stage);
      const int taddr_p_stage = taddr_s_stage + BLOCK_N / 2;

      float rowmax = -FLT_MAX;
      float rowsum = 0.0f;
      constexpr float LOG2_E = 1.4426950408889634f;
      const float softmax_scale_log2 = rsqrtf(float(HEAD_DIM)) * LOG2_E;
      for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        mbarrier::wait(mbar_qk_done_addr, phase_qk_done);
        phase_qk_done ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");


        // FA4's split_P_arrive is also a register-pressure optimization: it stores the first
        // 3/4 of P early. The default local path caches that prefix in C++ fragments; the asm
        // reread row engine instead avoids any resident prefix carry and repulls the row later.
        float2 s_head_pair[P_HEAD_N8][4];
        float tile_rowmax_raw = -FLT_MAX;
        {
#pragma unroll
          for (int n8 = 0; n8 < BLOCK_N / 8; ++n8) {
            float s8[8];
            const int taddr = taddr_s_stage + ((softmax_slab * 32) << 16) + n8 * 8;
            tcgen05::ld_32x32b_x8(taddr, s8);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              tile_rowmax_raw = fmaxf(tile_rowmax_raw, s8[i]);
            }
            if (n8 >= P_TAIL_N8) {
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                s_head_pair[n8 - P_TAIL_N8][j] = make_float2(s8[2 * j], s8[2 * j + 1]);
              }
            }
          }
        }

        const float tile_rowmax = tile_rowmax_raw * softmax_scale_log2;
        float new_rowmax = fmaxf(rowmax, tile_rowmax);
        float rescale = approx::fast_exp2(rowmax - new_rowmax);
        if (rowmax > -1.0e30f && new_rowmax > rowmax &&
            (new_rowmax - rowmax) < RESCALE_THRESHOLD_LOG2) {
          new_rowmax = rowmax;
          rescale = 1.0f;
        }
        rowmax = new_rowmax;
        rowsum *= rescale;
        rescale_smem[softmax_stage][softmax_row_tid] = rescale;

        __syncwarp();
        if (lane_id == 0) {
          mbarrier::arrive(mbar_stats_ready_addr);
        }

        float tile_rowsum = 0.0f;
        {
          const int trow = ((softmax_slab * 32) << 16);
          const int taddr_s_row = taddr_s_stage + trow;
          const int taddr_p_row = taddr_p_stage + trow;
          const float row_bias = -rowmax;
          uint32_t p8_packed[8];
          // TODO: `prefix_signaled` is likely redundant because the linear n16
          // loop reaches the prefix signal point only once.
          bool prefix_signaled = false;
          // int prefix_cols_stored = 0;
          float2 prefix_rowsum_pair = make_float2(0.0f, 0.0f);
#pragma unroll
          for (int n16 = BLOCK_N / 16 - 1; n16 >= 0; --n16) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int frag = n16 * 2 + half;
              float s8[8];
              if (frag >= P_TAIL_N8) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const float2 x_pair =
                      f32x2::fma_rn_ftz(s_head_pair[frag - P_TAIL_N8][j], softmax_scale_log2,
                                       row_bias);
                  s8[2 * j] = x_pair.x;
                  s8[2 * j + 1] = x_pair.y;
                }
              } else {
                tcgen05::ld_32x32b_x8(taddr_s_row + frag * 8, s8);
              }
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                float x0;
                float x1;
                if (frag >= P_TAIL_N8) {
                  x0 = s8[2 * j];
                  x1 = s8[2 * j + 1];
                } else {
                  const float2 x_pair =
                      f32x2::fma_rn_ftz(make_float2(s8[2 * j], s8[2 * j + 1]),
                                       softmax_scale_log2, row_bias);
                  x0 = x_pair.x;
                  x1 = x_pair.y;
                }
                const int col0 = frag * 8 + 2 * j;
                const int frg = col0 / E2E_FRG_TILE;
                const int k_pair = col0 % E2E_FRG_TILE;
                float p0;
                float p1;
                if ((k_pair % E2E_FREQ) >= (E2E_FREQ - E2E_RES) &&
                    frg < (BLOCK_N / E2E_FRG_TILE) - E2E_FRG_LIMIT) {
                  approx::e2e_exp2_pair(x0, x1, p0, p1);
                } else {
                  p0 = approx::fast_exp2(x0);
                  p1 = approx::fast_exp2(x1);
                }
                if (frag >= P_TAIL_N8) {
                  prefix_rowsum_pair = f32x2::add_rn_ftz(prefix_rowsum_pair, make_float2(p0, p1));
                } else {
                  tile_rowsum += p0 + p1;
                }
                p8_packed[half * 4 + j] = bf16::pack2_to_u32(p0, p1);
              }
            }
            tcgen05::st_32x32b_x8_u32(taddr_p_row + n16 * 8, p8_packed);
            // if (n16 >= P_SPLIT_N16) {
            //   prefix_cols_stored += 16;
            // }
            if (n16 == P_SPLIT_N16 && !prefix_signaled) {
              asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
              asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
              __syncwarp();
              if (lane_id == 0) {
                mbarrier::arrive(mbar_p_half_ready_addr);
              }
              prefix_signaled = true;
              tile_rowsum += prefix_rowsum_pair.x + prefix_rowsum_pair.y;
              prefix_rowsum_pair = make_float2(0.0f, 0.0f);
            }
          }
          asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
        }

        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        __syncwarp();
        if (lane_id == 0) {
          mbarrier::arrive(mbar_p_ready_addr);
        }

        rowsum += tile_rowsum;
      }
      const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, softmax_stage);
      const int final_pv_phase = (kv_tiles - 1) & 1;
      mbarrier::wait(mbar_pv_done_addr, final_pv_phase);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      const float inv_denom = 1.0f / rowsum;
      const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, softmax_stage);
      const int q_row0_stage = q_stage::row0(q_row0_base, softmax_stage);
      float o8[8];
#pragma unroll
      for (int d8 = 0; d8 < HEAD_DIM / 8; ++d8) {
        const int taddr = taddr_o_stage + ((softmax_slab * 32) << 16) + d8 * 8;
        tcgen05::ld_32x32b_x8(taddr, o8);

        nv_bfloat162 out_bf16x2[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 v = make_float2(o8[2 * i] * inv_denom, o8[2 * i + 1] * inv_denom);
          out_bf16x2[i] = __float22bfloat162_rn(v);
        }

        nv_bfloat16* out_ptr = O_ptr + (q_row0_stage + softmax_row_tid) * HEAD_DIM + d8 * 8;
        reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out_bf16x2)[0];
      }

      if (lane_id == 0) {
        mbarrier::arrive(mbar_work_done_addr);
      }
    }
  }

  // Correction warps: shared across both Q stages.
  if (is_corr_warp) {
    // For a given Q stage, correction warp k reads the rescale factors
    // written by softmax slab k for that same 32-row range.
    const int corr_warp = warp_id - CORR_WARP_BASE;

    for (int bid = blockIdx.x, work_iter = 0; bid < total_work_tiles;
         bid += gridDim.x, ++work_iter) {

      mbarrier::wait(mbar_setup_done_addr, work_iter & 1);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      int phase_stats_ready[q_stage::STAGES] = {0, 0};
      for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const bool first_kv = (kv_tile == 0);
        const int row0 = corr_warp * WARP_SIZE;
        for (int q = 0; q < q_stage::STAGES; ++q) {
          const int mbar_stats_ready_addr = q_stage::mbar_addr(mbar_stats_ready_base, q);
          const int mbar_p_half_ready_addr = q_stage::mbar_addr(mbar_p_half_ready_base, q);
          const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);

          // stats_ready also carries the previous PV/O TMEM dependency into correction;
          // bridge the mbarrier wait back into tcgen05 before correction's TMEM ld/st.
          mbarrier::wait(mbar_stats_ready_addr, phase_stats_ready[q]);
          phase_stats_ready[q] ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
          __syncwarp();

          if (!first_kv) {
            const float row_rescale = rescale_smem[q][row0 + lane_id];
            const unsigned need_mask = __ballot_sync(__activemask(), row_rescale < 1.0f);
            if (need_mask != 0) {
              float o8[8];
#pragma unroll
              for (int d8 = 0; d8 < HEAD_DIM / 8; ++d8) {
                const int taddr = taddr_o_stage + ((corr_warp * 32) << 16) + d8 * 8;
                tcgen05::ld_32x32b_x8(taddr, o8);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                  o8[i] *= row_rescale;
                }
                tcgen05::st_32x32b_x8(taddr, o8);
              }
              asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
              asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
            }
          }

          __syncwarp();
          if (lane_id == 0) {
            mbarrier::arrive(mbar_p_half_ready_addr);
          }
        }
      }

      __syncwarp();
      if (lane_id == 0) {
        mbarrier::arrive(mbar_work_done_addr);
      }
    }
  }

  if (is_mma_warp) {
    const int final_work_phase = (work_count - 1) & 1;
    mbarrier::wait(mbar_work_done_addr, final_work_phase);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(taddr_base), "r"(TMEM_ALLOC_COLS)
                 : "memory");
  }
}

void attention_tcgen05_v14_launch_impl(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
                                       const nv_bfloat16* V_ptr, nv_bfloat16* O_ptr, int bs,
                                       int len_q, int len_kv, cudaStream_t stream) {
  TORCH_CHECK(bs > 0, "bs must be positive");
  TORCH_CHECK(len_q >= q_stage::STAGES * BLOCK_M, "len_q must be at least 256");
  TORCH_CHECK(len_q % (q_stage::STAGES * BLOCK_M) == 0, "len_q must be a multiple of 256");
  TORCH_CHECK(len_kv >= BLOCK_N, "len_kv must be at least 128");
  TORCH_CHECK(len_kv % BLOCK_N == 0, "len_kv must be a multiple of 128");

  const int Q_height = bs * len_q;
  const int KV_height = bs * len_kv;

  CUtensorMap Q_tmap{};
  CUtensorMap K_tmap{};
  CUtensorMap V_tmap{};

  init_tmap_2d_simple(&Q_tmap, const_cast<nv_bfloat16*>(Q_ptr), /*global_height=*/Q_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_M, /*shared_width=*/SW128_ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_128B);

  init_tmap_2d_simple(&K_tmap, const_cast<nv_bfloat16*>(K_ptr), /*global_height=*/KV_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_N, /*shared_width=*/SW128_ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_128B);

  init_tmap_2d_simple(&V_tmap, const_cast<nv_bfloat16*>(V_ptr), /*global_height=*/KV_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/SW128_ATOM_ROWS, /*shared_width=*/SW128_ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_128B);

  const int total_work_tiles = bs * (len_q / (q_stage::STAGES * BLOCK_M));

  int device = 0;
  int sm_count = 0;
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  const int persistent_grid = sm_count * PERSISTENT_CTAS_PER_SM;
  const int grid = (total_work_tiles < persistent_grid) ? total_work_tiles : persistent_grid;
  const size_t smem_bytes =
      size_t(q_stage::STAGES * Q_TILE_BYTES + kv_ring::STAGES * KV_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v14_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v14_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv, total_work_tiles);
  check_cuda(cudaGetLastError());
}

void attention_tcgen05_v14_launch(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
                                  const nv_bfloat16* V_ptr, nv_bfloat16* O_ptr, int bs,
                                  int len_q, int len_kv, cudaStream_t stream) {
  attention_tcgen05_v14_launch_impl(Q_ptr, K_ptr, V_ptr, O_ptr, bs, len_q, len_kv, stream);
}
