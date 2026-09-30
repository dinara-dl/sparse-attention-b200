#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>

void fp4_tile_topk_launch(
    const float* raw_scores, int32_t* selected_indices, int batch_size,
    int num_kv_heads, int len_q, int kv_tiles, cudaStream_t stream);

namespace {

void validate_scores(const at::Tensor& scores, int64_t batch_size, int64_t len_q) {
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(scores.scalar_type() == at::kFloat,
              "scores must have float32 dtype");
  TORCH_CHECK(scores.dim() == 3 && scores.is_contiguous(),
              "scores must be contiguous [Hkv, KV_tiles, B*Sq]");
  TORCH_CHECK(batch_size > 0 && batch_size <= INT_MAX,
              "batch_size must be positive and fit int32");
  TORCH_CHECK(len_q > 0 && len_q <= INT_MAX && len_q % 128 == 0,
              "len_q must be a positive multiple of 128 and fit int32");
  TORCH_CHECK(scores.size(0) > 0 && scores.size(0) <= INT_MAX,
              "Hkv must be positive and fit int32");
  TORCH_CHECK(scores.size(1) >= 16 && scores.size(1) <= 128,
              "expected 16 to 128 KV blocks");
  TORCH_CHECK(batch_size <= INT_MAX / len_q,
              "B*Sq must fit int32");
  TORCH_CHECK(scores.size(2) == batch_size * len_q,
              "scores query extent must equal batch_size * len_q");
  const int64_t head_batches = batch_size * scores.size(0);
  TORCH_CHECK(head_batches <= INT_MAX / (len_q / 128),
              "CTA count must fit int32");
}

void launch(const at::Tensor& scores, const at::Tensor& selected,
            int64_t batch_size, int64_t len_q) {
  const c10::cuda::CUDAGuard device_guard(scores.device());
  fp4_tile_topk_launch(
      scores.data_ptr<float>(), selected.data_ptr<int32_t>(),
      static_cast<int>(batch_size), static_cast<int>(scores.size(0)),
      static_cast<int>(len_q), static_cast<int>(scores.size(1)),
      at::cuda::getCurrentCUDAStream().stream());
}

at::Tensor tile_topk(const at::Tensor& scores, int64_t batch_size, int64_t len_q) {
  validate_scores(scores, batch_size, len_q);
  auto selected = at::empty(
      {batch_size, scores.size(0), len_q / 128, 16},
      scores.options().dtype(at::kInt));
  launch(scores, selected, batch_size, len_q);
  return selected;
}

void tile_topk_out(const at::Tensor& scores, const at::Tensor& selected,
                   int64_t batch_size, int64_t len_q) {
  validate_scores(scores, batch_size, len_q);
  TORCH_CHECK(selected.is_cuda() && selected.device() == scores.device(),
              "selected must be on the same CUDA device as scores");
  TORCH_CHECK(selected.scalar_type() == at::kInt && selected.is_contiguous(),
              "selected must be contiguous int32");
  TORCH_CHECK(selected.sizes() == at::IntArrayRef(
                  {batch_size, scores.size(0), len_q / 128, 16}),
              "selected must have shape [B, Hkv, Sq/128, 16]");
  TORCH_CHECK(!selected.is_alias_of(scores), "selected must not alias scores");
  launch(scores, selected, batch_size, len_q);
}

}  // namespace

TORCH_LIBRARY(xomtention_tile_topk, m) {
  m.def("tile_topk(Tensor scores, int batch_size, int len_q) -> Tensor");
  m.impl("tile_topk", &tile_topk);
  m.def("tile_topk_out(Tensor scores, Tensor(a!) selected, int batch_size, "
        "int len_q) -> ()");
  m.impl("tile_topk_out", &tile_topk_out);
}
