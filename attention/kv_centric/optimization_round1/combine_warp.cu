// Candidate C1: one warp per output row, four output rows per CTA.
// Same FP32 normalized partial O + natural-log LSE contract as kv0.
// No shared memory or CTA barrier: every shuffle operates within one full warp.
#include "../primitives.cuh"
#include <math_constants.h>
#include <cmath>

namespace kv_combine_warp {
__global__ __launch_bounds__(128) void kernel(const float* __restrict__ partial,
    const float* __restrict__ lses, nv_bfloat16* __restrict__ out,
    float* __restrict__ out_lse, int rows) {
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  // Uniform for each warp; the other warps never wait at a CTA barrier.
  if (row >= rows) return;
  constexpr unsigned mask = 0xffffffffU;
  const float own_lse = lane < 16 ? lses[int64_t(row) * 16 + lane] : -CUDART_INF_F;
  float maximum = own_lse;
  #pragma unroll
  for (int distance = 16; distance > 0; distance /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, distance));
  const float own_weight = lane < 16 ? expf(own_lse - maximum) : 0.0f;
  float denom = own_weight;
  #pragma unroll
  for (int distance = 16; distance > 0; distance /= 2)
    denom += __shfl_xor_sync(mask, denom, distance);

  float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
  #pragma unroll
  for (int rank = 0; rank < 16; ++rank) {
    const float weight = __shfl_sync(mask, own_weight, rank);
    // Each lane owns four adjacent features; every source vector is 16B aligned.
    const float4 value = *reinterpret_cast<const float4*>(
        partial + (int64_t(row) * 16 + rank) * 128 + lane * 4);
    a0 = fmaf(weight, value.x, a0);
    a1 = fmaf(weight, value.y, a1);
    a2 = fmaf(weight, value.z, a2);
    a3 = fmaf(weight, value.w, a3);
  }
  const uint2 packed = {kv0::pack2(a0 / denom, a1 / denom),
                        kv0::pack2(a2 / denom, a3 / denom)};
  *reinterpret_cast<uint2*>(out + int64_t(row) * 128 + lane * 4) = packed;
  if (lane == 0) out_lse[row] = maximum + logf(denom);
}
}

void kv1_launch_combine(const float* partial, const float* lses, nv_bfloat16* out,
                        float* out_lse, int rows, cudaStream_t stream) {
  kv_combine_warp::kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      partial, lses, out, out_lse, rows);
  kv0::cuda_check(cudaGetLastError());
}
