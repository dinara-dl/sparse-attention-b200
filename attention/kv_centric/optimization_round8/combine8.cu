// Round8 Combine for BF16 partial O. Same one-warp-per-row structure, same
// FP32 LSE weights and the same accumulation order as round1's combine_warp,
// so out_lse is bit-identical to the FP32-partials path; only the partial
// operands are BF16 (8 bytes per lane per rank instead of 16).
#include "../primitives.cuh"
#include <math_constants.h>
#include <cmath>

namespace kv_combine8_bf16 {
__global__ __launch_bounds__(128) void kernel(const nv_bfloat16* __restrict__ partial,
    const float* __restrict__ lses, nv_bfloat16* __restrict__ out,
    float* __restrict__ out_lse, int rows) {
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
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
    // Each lane owns four adjacent features: one 8-byte BF16x4 load per rank.
    const uint2 raw = *reinterpret_cast<const uint2*>(
        partial + (int64_t(row) * 16 + rank) * 128 + lane * 4);
    union { uint32_t bits; nv_bfloat162 bf; } lo, hi;
    lo.bits = raw.x; hi.bits = raw.y;
    const float2 v01 = __bfloat1622float2(lo.bf), v23 = __bfloat1622float2(hi.bf);
    a0 = fmaf(weight, v01.x, a0);
    a1 = fmaf(weight, v01.y, a1);
    a2 = fmaf(weight, v23.x, a2);
    a3 = fmaf(weight, v23.y, a3);
  }
  const uint2 packed = {kv0::pack2(a0 / denom, a1 / denom),
                        kv0::pack2(a2 / denom, a3 / denom)};
  *reinterpret_cast<uint2*>(out + int64_t(row) * 128 + lane * 4) = packed;
  if (lane == 0) out_lse[row] = maximum + logf(denom);
}
}

void kv8_launch_combine_bf16(const nv_bfloat16* partial, const float* lses, nv_bfloat16* out,
                             float* out_lse, int rows, cudaStream_t stream) {
  kv_combine8_bf16::kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      partial, lses, out, out_lse, rows);
  kv0::cuda_check(cudaGetLastError());
}
