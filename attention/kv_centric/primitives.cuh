#pragma once
// Minimal SM100 helpers adapted from the existing project's common.cuh and
// smem_layout_swizzle128.cuh (hashes in baseline_manifest.json).
// These PTX/layout primitives are inherited, not new KV-centric optimizations.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/util/Exception.h>
#include <cstdint>

namespace kv0 {
constexpr int M = 128, N = 128, D = 128, TOPK = 16;
constexpr int TILE_BYTES = 128 * 128 * 2;

inline void cuda_check(cudaError_t error) {
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
inline void cu_check(CUresult error) {
  if (error != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(error, &message);
    TORCH_CHECK(false, message ? message : "CUDA driver failure");
  }
}

inline CUtensorMap make_tmap(const nv_bfloat16* ptr, int height, int tile_height) {
  CUtensorMap map{};
  uint64_t dims[] = {D, static_cast<uint64_t>(height)};
  uint64_t strides[] = {D * sizeof(nv_bfloat16)};
  uint32_t box[] = {64, static_cast<uint32_t>(tile_height)};
  uint32_t steps[] = {1, 1};
  cu_check(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      const_cast<nv_bfloat16*>(ptr), dims, strides, box, steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

__device__ __forceinline__ int smem_address(const void* ptr) {
  return static_cast<int>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void barrier_init(int addr) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void barrier_wait(int addr) {
  // Every barrier in v0 is one-shot, with initial parity zero.
  asm volatile("{ .reg .pred p; WAIT: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], 0; "
               "@!p bra WAIT; }" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void expect_bytes(int addr, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mma_commit(int addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(addr) : "memory");
}
__device__ __forceinline__ void tma_copy(int dst, const CUtensorMap* map,
                                       int x, int y, int barrier) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(map), "r"(x), "r"(y), "r"(barrier) : "memory");
}
__device__ __forceinline__ uint64_t encode(uint64_t bytes) {
  return (bytes & 0x3ffffULL) >> 4;
}
__device__ __forceinline__ uint64_t kmajor(int addr) {
  return encode(addr) | (encode(1024) << 32) | (1ULL << 46) | (2ULL << 61);
}
__device__ __forceinline__ uint64_t mnmajor(int addr) {
  return encode(addr) | (encode(1024) << 16) | (encode(2048) << 32)
         | (1ULL << 46) | (2ULL << 61);
}
__device__ __forceinline__ void mma_ss(int dest, uint64_t a, uint64_t b,
                                      uint32_t desc, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
               :: "r"(dest), "l"(a), "l"(b), "r"(desc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void mma_ts(int dest, int a, uint64_t b,
                                      uint32_t desc, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p; }"
               :: "r"(dest), "r"(a), "l"(b), "r"(desc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void load8(int addr, float (&v)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                 "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
               : "r"(addr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void store8(int addr, const uint32_t (&v)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 "
               "[%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]),
                  "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]) : "memory");
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  union { nv_bfloat162 bf; uint32_t bits; } value;
  value.bf = __floats2bfloat162_rn(a, b);
  return value.bits;
}
}  // namespace kv0
