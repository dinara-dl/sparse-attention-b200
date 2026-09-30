#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>

#define SPARSE_DECL(NAME) void NAME(const nv_bfloat16*, const nv_bfloat16*, \
    const nv_bfloat16*, const int32_t*, nv_bfloat16*, int, int, int, int, int, int, cudaStream_t)
SPARSE_DECL(dense13_sparse_two_q_launch);
SPARSE_DECL(dense14_sparse_two_q_launch);
SPARSE_DECL(dense14_sparse_gqa_launch);

namespace {

void progression_out(const at::Tensor& q, const at::Tensor& k,
                              const at::Tensor& v,
                              const at::Tensor& selected_indices,
                              const at::Tensor& output, int64_t variant) {
  TORCH_CHECK(variant >= 0 && variant <= 2, "variant must be 0, 1 or 2");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "q, k, and v must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "q, k, and v must be on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16,
              "q, k, and v must be bfloat16");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "This attention implementation is forward-only");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q, k, and v must be contiguous");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q must be [B,Hq,Sq,128], k/v must be [B,Hkv,Skv,128]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, and v batch dimensions must match");
  TORCH_CHECK(k.size(1) == v.size(1),
              "k and v head dimensions must match");
  TORCH_CHECK(q.size(0) > 0 && k.size(1) > 0,
              "batch and KV heads must be positive");
  TORCH_CHECK(q.size(1) >= k.size(1) && q.size(1) % k.size(1) == 0,
              "Hq must be a multiple of and at least Hkv");
  const int64_t group_size = q.size(1) / k.size(1);
  TORCH_CHECK(group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8 ||
                  group_size == 16,
              "supported ratios Hq/Hkv are 1, 2, 4, 8, and 16");
  TORCH_CHECK(variant != 2 || group_size >= 2, "head pairing requires G>=2");
  TORCH_CHECK(variant == 2 || q.size(2) % 256 == 0, "Q-position pairing requires Sq%256==0");
  TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && v.size(3) == 128,
              "head dimension must be 128");
  TORCH_CHECK(k.size(2) == v.size(2),
              "k and v sequence lengths must match");
  TORCH_CHECK(q.size(2) >= 128 && q.size(2) % 128 == 0,
              "Sq must be a positive multiple of 128");
  TORCH_CHECK(k.size(2) >= 128 && k.size(2) % 128 == 0,
              "Skv must be a positive multiple of 128");

  TORCH_CHECK(selected_indices.is_cuda(),
              "selected_indices must be a CUDA tensor");
  TORCH_CHECK(selected_indices.device() == q.device(),
              "selected_indices must be on the same CUDA device as q");
  TORCH_CHECK(selected_indices.scalar_type() == at::kInt,
              "selected_indices must have int32 dtype");
  TORCH_CHECK(selected_indices.is_contiguous(),
              "selected_indices must be contiguous");
  TORCH_CHECK(selected_indices.dim() == 4,
              "selected_indices must be [B,Hkv,Sq/128,TopK]");
  TORCH_CHECK(selected_indices.size(0) == q.size(0) &&
                  selected_indices.size(1) == k.size(1),
              "selected_indices batch/KV-head dimensions must match q/k");
  TORCH_CHECK(selected_indices.size(2) == q.size(2) / 128,
              "selected_indices q-tile dimension must equal Sq/128");
  TORCH_CHECK(selected_indices.numel() <= INT_MAX, "selected offsets must fit int32");
  const int selected_count = static_cast<int>(selected_indices.size(3));
  TORCH_CHECK(selected_count > 0,
              "selected_count must be positive");
  TORCH_CHECK(selected_count <= k.size(2) / 128,
              "selected_count cannot exceed the number of KV tiles");

  TORCH_CHECK(q.numel() <= INT_MAX &&
                  k.numel() <= INT_MAX,
              "Q/KV element offsets must fit int32");
  TORCH_CHECK(output.is_cuda() && output.device() == q.device() &&
                  output.scalar_type() == q.scalar_type() &&
                  output.sizes() == q.sizes() && output.is_contiguous(),
              "output must match q shape, dtype, device, and contiguous layout");
  TORCH_CHECK(!output.is_alias_of(q) && !output.is_alias_of(k) &&
                  !output.is_alias_of(v) && !output.is_alias_of(selected_indices),
              "output must not alias inputs");

  // ID range and uniqueness are the caller's contract, checked by the harness.
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto launch = variant == 0 ? dense13_sparse_two_q_launch
                : variant == 1 ? dense14_sparse_two_q_launch
                               : dense14_sparse_gqa_launch;
  launch(
      reinterpret_cast<const nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(v.data_ptr()),
      selected_indices.data_ptr<int32_t>(),
      reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
      static_cast<int>(k.size(1)), static_cast<int>(q.size(2)),
      static_cast<int>(k.size(2)), selected_count,
      at::cuda::getCurrentCUDAStream().stream());
}

at::Tensor progression_attention(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& ids, int64_t variant) {
  auto output = at::empty_like(q);
  progression_out(q, k, v, ids, output, variant);
  return output;
}

}  // namespace

TORCH_LIBRARY(xomtention_progression, m) {
  m.def("attention(Tensor q, Tensor k, Tensor v, Tensor ids, int variant) -> Tensor");
  m.impl("attention", &progression_attention);
  m.def("attention_out(Tensor q, Tensor k, Tensor v, Tensor ids, Tensor(a!) output, int variant) -> ()");
  m.impl("attention_out", &progression_out);
}
