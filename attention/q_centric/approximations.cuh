#pragma once

#include <cuda_runtime.h>

namespace approx {

// Hardware exp2 approximation helper introduced by kernel 11:

__device__ __forceinline__ float fast_exp2(float x) {
  float out;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}

// Software exp2 approximation helpers introduced by kernel 12:

__device__ __forceinline__ float2 fma_rn_ftz_f32x2(float2 x, float2 a, float b) {
  float2 out;
  asm volatile(
      "{\n\t"
      ".reg .b64 vx, va, vb, vo;\n\t"
      "mov.b64 vx, {%2, %3};\n\t"
      "mov.b64 va, {%4, %5};\n\t"
      "mov.b64 vb, {%6, %6};\n\t"
      "fma.rn.ftz.f32x2 vo, vx, va, vb;\n\t"
      "mov.b64 {%0, %1}, vo;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(x.x), "f"(x.y), "f"(a.x), "f"(a.y), "f"(b));
  return out;
}

__device__ __forceinline__ float2 add_rm_ftz_f32x2(float2 a, float2 b) {
  float2 out;
  asm volatile(
      "{\n\t"
      ".reg .b64 va, vb, vo;\n\t"
      "mov.b64 va, {%2, %3};\n\t"
      "mov.b64 vb, {%4, %5};\n\t"
      "add.rm.ftz.f32x2 vo, va, vb;\n\t"
      "mov.b64 {%0, %1}, vo;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return out;
}

__device__ __forceinline__ float2 sub_rn_ftz_f32x2(float2 a, float2 b) {
  float2 out;
  asm volatile(
      "{\n\t"
      ".reg .b64 va, vb, vo;\n\t"
      "mov.b64 va, {%2, %3};\n\t"
      "mov.b64 vb, {%4, %5};\n\t"
      "sub.rn.ftz.f32x2 vo, va, vb;\n\t"
      "mov.b64 {%0, %1}, vo;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return out;
}

__device__ __forceinline__ float combine_int_frac_ex2(float rounded, float frac_ex2) {
  float out;
  asm volatile(
      "{\n\t"
      ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
      "mov.b32 x_rounded_i, %1;\n\t"
      "mov.b32 frac_ex_i, %2;\n\t"
      "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
      "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
      "mov.b32 %0, out_i;\n\t"
      "}\n"
      : "=f"(out)
      : "f"(rounded), "f"(frac_ex2));
  return out;
}

__device__ __forceinline__ void e2e_exp2_pair(float x, float y, float& out0, float& out1) {
  constexpr float ROUND_INT = 12582912.0f;  // 2^23 + 2^22, official FA4 rounding trick.
  float2 clamped = make_float2(fmaxf(x, -127.0f), fmaxf(y, -127.0f));
  const float2 round_pair = make_float2(ROUND_INT, ROUND_INT);
  const float2 rounded = add_rm_ftz_f32x2(clamped, round_pair);
  const float2 rounded_back = sub_rn_ftz_f32x2(rounded, round_pair);
  const float2 frac = sub_rn_ftz_f32x2(clamped, rounded_back);

  float2 poly = make_float2(__uint_as_float(0x3D9DF09D), __uint_as_float(0x3D9DF09D));
  poly = fma_rn_ftz_f32x2(poly, frac, __uint_as_float(0x3E6906A4));
  poly = fma_rn_ftz_f32x2(poly, frac, __uint_as_float(0x3F31F519));
  poly = fma_rn_ftz_f32x2(poly, frac, 1.0f);

  out0 = combine_int_frac_ex2(rounded.x, poly.x);
  out1 = combine_int_frac_ex2(rounded.y, poly.y);
}

}  // namespace approx
