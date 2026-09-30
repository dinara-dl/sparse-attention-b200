// Round11 Combine for token-major partial O [B,Sq,16,Hq,D] (FP32 or BF16).
// Same warp-per-row structure, FP32 LSE weights and accumulation order as
// round1/round8, so final O and LSE are bit-identical to the [B,Hq,Sq,16,D]
// Combine; only the partial address changes. LSE stays [B,Hq,Sq,16].
#include "../primitives.cuh"
#include <math_constants.h>
#include <cmath>
#include <type_traits>

namespace kv_combine11 {
template<typename T>
__device__ __forceinline__ void load4(const T* src,float& a,float& b,float& c,float& d) {
  if constexpr(std::is_same<T,float>::value) {
    const float4 v=*reinterpret_cast<const float4*>(src);a=v.x;b=v.y;c=v.z;d=v.w;
  } else {
    const uint2 raw=*reinterpret_cast<const uint2*>(src);
    union { uint32_t bits; nv_bfloat162 bf; } lo,hi;
    lo.bits=raw.x;hi.bits=raw.y;
    const float2 v01=__bfloat1622float2(lo.bf),v23=__bfloat1622float2(hi.bf);
    a=v01.x;b=v01.y;c=v23.x;d=v23.y;
  }
}
template<typename T>
__global__ __launch_bounds__(128) void kernel(const T* __restrict__ partial,
    const float* __restrict__ lses, nv_bfloat16* __restrict__ out,
    float* __restrict__ out_lse, int rows, int hq, int sq) {
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

  // row = (b*hq + h)*sq + s  ->  token-major partial base ((b*sq + s)*16 + rank)*hq + h
  const int s = row % sq, bh = row / sq, h = bh % hq, b = bh / hq;
  const int64_t base = (int64_t(b) * sq + s) * 16;
  float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
  #pragma unroll
  for (int rank = 0; rank < 16; ++rank) {
    const float weight = __shfl_sync(mask, own_weight, rank);
    float v0,v1,v2,v3;
    load4(partial + ((base + rank) * hq + h) * 128 + lane * 4, v0,v1,v2,v3);
    a0 = fmaf(weight, v0, a0);
    a1 = fmaf(weight, v1, a1);
    a2 = fmaf(weight, v2, a2);
    a3 = fmaf(weight, v3, a3);
  }
  const uint2 packed = {kv0::pack2(a0 / denom, a1 / denom),
                        kv0::pack2(a2 / denom, a3 / denom)};
  *reinterpret_cast<uint2*>(out + int64_t(row) * 128 + lane * 4) = packed;
  if (lane == 0) out_lse[row] = maximum + logf(denom);
}
}

void kv11_launch_combine_f32(const float* partial, const float* lses, nv_bfloat16* out,
                             float* out_lse, int rows, int hq, int sq, cudaStream_t stream) {
  kv_combine11::kernel<float><<<(rows + 3) / 4, 128, 0, stream>>>(partial, lses, out, out_lse, rows, hq, sq);
  kv0::cuda_check(cudaGetLastError());
}
void kv11_launch_combine_bf16(const nv_bfloat16* partial, const float* lses, nv_bfloat16* out,
                              float* out_lse, int rows, int hq, int sq, cudaStream_t stream) {
  kv_combine11::kernel<nv_bfloat16><<<(rows + 3) / 4, 128, 0, stream>>>(partial, lses, out, out_lse, rows, hq, sq);
  kv0::cuda_check(cudaGetLastError());
}
