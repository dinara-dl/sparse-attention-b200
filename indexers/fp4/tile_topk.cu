// Tile-level adapter for the MSA-inspired FP4 indexer.
// Ported from codex_1/post_v14/04_fp4_mha_indexer/tile_topk.cu.
//
// Input:  contiguous finite FP32 scores [Hkv, KV_tiles, B * Sq].
//         The scorer has already reduced over keys inside each KV block.
// Output: contiguous int32 IDs [B, Hkv, Sq / 128, 16], local to each sequence.
// One CTA reduces over 128 queries, then selects TopK16 KV blocks.
// This adapter does not perform FP4 matmul or attention.

#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kQBlock = 128;
constexpr int kTopK = 16;
constexpr int kMaxKvTiles = 128;
constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;

__device__ __forceinline__ void warp_argmax(float& score, int& index) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_score = __shfl_down_sync(0xffffffffu, score, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_score > score ||
        (other_score == score && other_index < index)) {
      score = other_score;
      index = other_index;
    }
  }
}

__global__ void tile_max_topk16_kernel(
    const float* __restrict__ raw_scores,
    int32_t* __restrict__ selected_indices,
    int batch_size,
    int num_kv_heads,
    int len_q,
    int kv_tiles) {
  const int row = static_cast<int>(blockIdx.x);
  const int q_tiles = len_q / kQBlock;
  const int q_tile = row % q_tiles;
  const int head_batch = row / q_tiles;
  const int head = head_batch % num_kv_heads;
  const int batch = head_batch / num_kv_heads;
  const int total_q = batch_size * len_q;
  const int q_begin = batch * len_q + q_tile * kQBlock;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  __shared__ float tile_scores[kMaxKvTiles];

  // Four warps reduce four KV blocks concurrently. Each lane reads four
  // query scores; warp reduction produces one score for the whole Q tile.
  for (int group = 0; group < (kv_tiles + kWarps - 1) / kWarps; ++group) {
    const int kv_tile = group * kWarps + warp;
    if (kv_tile < kv_tiles) {
      float local_max = -FLT_MAX;
#pragma unroll
      for (int q_offset = lane; q_offset < kQBlock; q_offset += 32) {
        const int64_t offset =
            (static_cast<int64_t>(head) * kv_tiles + kv_tile) * total_q +
            q_begin + q_offset;
        local_max = fmaxf(local_max, raw_scores[offset]);
      }
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        local_max =
            fmaxf(local_max,
                  __shfl_down_sync(0xffffffffu, local_max, delta));
      }
      if (lane == 0) {
        tile_scores[kv_tile] = local_max;
      }
    }
  }
  __syncthreads();

  // Warp 0 holds up to four candidates per lane. Repeated argmax emits
  // score-descending IDs; the smaller block ID wins a tie.
  if (warp == 0) {
    float lane_scores[4];
    int lane_indices[4];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      const int index = lane + slot * 32;
      lane_indices[slot] = index < kv_tiles ? index : INT_MAX;
      lane_scores[slot] =
          index < kv_tiles ? tile_scores[index] : -FLT_MAX;
    }

#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      float best_score = -FLT_MAX;
      int best_index = INT_MAX;
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        if (lane_scores[slot] > best_score ||
            (lane_scores[slot] == best_score &&
             lane_indices[slot] < best_index)) {
          best_score = lane_scores[slot];
          best_index = lane_indices[slot];
        }
      }
      warp_argmax(best_score, best_index);
      best_index = __shfl_sync(0xffffffffu, best_index, 0);
      if (lane == 0) {
        selected_indices[static_cast<int64_t>(row) * kTopK + rank] =
            best_index;
      }
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        if (lane_indices[slot] == best_index) {
          lane_scores[slot] = -FLT_MAX;
          // Also retire the ID so even a valid -FLT_MAX score is picked once.
          lane_indices[slot] = INT_MAX;
        }
      }
    }
  }
}

}  // namespace

void fp4_tile_topk_launch(
    const float* raw_scores,
    int32_t* selected_indices,
    int batch_size,
    int num_kv_heads,
    int len_q,
    int kv_tiles,
    cudaStream_t stream) {
  TORCH_CHECK(batch_size > 0 && num_kv_heads > 0,
              "batch and KV heads must be positive");
  TORCH_CHECK(len_q > 0 && len_q % kQBlock == 0,
              "Sq must be a positive multiple of 128");
  TORCH_CHECK(kv_tiles >= kTopK && kv_tiles <= kMaxKvTiles,
              "tile TopK16 supports 16 to 128 KV blocks");
  const int64_t head_batches = static_cast<int64_t>(batch_size) * num_kv_heads;
  TORCH_CHECK(static_cast<int64_t>(batch_size) * len_q <= INT_MAX &&
                  head_batches <= INT_MAX / (len_q / kQBlock),
              "query and CTA counts must fit int32");
  const int rows = static_cast<int>(head_batches * (len_q / kQBlock));
  tile_max_topk16_kernel<<<rows, kThreads, 0, stream>>>(
      raw_scores, selected_indices, batch_size, num_kv_heads, len_q, kv_tiles);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "tile_max_topk16_kernel launch failed: ", cudaGetErrorString(status));
}
