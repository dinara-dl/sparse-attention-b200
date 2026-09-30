// Round11 preallocated API. -2 round9 split8 and -1 round9 split8_bf16 are the
// unchanged references (Q [B,Hq,Sq,D], partial [B,Hq,Sq,16,D]) and the exact
// parents of tma8 (FP32) and tma8_bf16/tma16_bf16 (BF16), which take Q as
// [B,Sq,Hq,D] and write partial O as [B,Sq,16,Hq,D]. LSE stays [B,Hq,Sq,16].
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <climits>
#include <algorithm>

void kv9_launch(const nv_bfloat16*,const nv_bfloat16*,const nv_bfloat16*,
    const int32_t*,const int32_t*,const int32_t*,void*,float*,
    int,int,int,int,int,int,int,int,cudaStream_t);
void kv11_launch(const nv_bfloat16*,const nv_bfloat16*,const nv_bfloat16*,
    const int32_t*,const int32_t*,const int32_t*,void*,float*,
    int,int,int,int,int,int,int,int,cudaStream_t);
void kv1_launch_combine(const float*,const float*,nv_bfloat16*,float*,int,cudaStream_t);
void kv11_launch_combine_f32(const float*,const float*,nv_bfloat16*,float*,int,int,int,cudaStream_t);
void kv11_launch_combine_bf16(const nv_bfloat16*,const float*,nv_bfloat16*,float*,int,int,int,cudaStream_t);
void kv8_launch_combine_bf16(const nv_bfloat16*,const float*,nv_bfloat16*,float*,int,cudaStream_t);
int64_t kv4_metadata_temp_bytes(int);
void kv4_build_metadata(const int32_t*,int32_t*,int32_t*,int32_t*,int32_t*,int32_t*,
    int32_t*,int32_t*,int32_t*,int32_t*,void*,size_t,
    int,int,int,int,int,int,cudaStream_t);

namespace {
void check(const at::Tensor& t,const at::Tensor& anchor,at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.device()==anchor.device(),"same CUDA device required");
  TORCH_CHECK(t.scalar_type()==dtype && t.is_contiguous(),"dtype/layout mismatch");
  TORCH_CHECK(!t.requires_grad(),"forward only");
}
int64_t metadata_temp_bytes(int64_t rows) {
  TORCH_CHECK(rows>0 && rows<INT_MAX-1,"metadata row count");
  return kv4_metadata_temp_bytes(rows);
}
void metadata_out(const at::Tensor& selected,const at::Tensor& counts,const at::Tensor& row_ptr,
    const at::Tensor& cursors,const at::Tensor& qids,const at::Tensor& ranks,
    const at::Tensor& chunks,const at::Tensor& task_ptr,const at::Tensor& tasks,
    const at::Tensor& error,const at::Tensor& temp,int64_t kv_blocks,int64_t g,int64_t chunk_size) {
  for(const auto& t:{selected,counts,row_ptr,cursors,qids,ranks,chunks,task_ptr,tasks,error})check(t,selected,at::kInt);
  check(temp,selected,at::kByte);
  TORCH_CHECK(selected.dim()==4 && selected.size(3)==16,"selected [B,Hkv,Sq,16]");
  const int64_t groups=selected.size(0)*selected.size(1),sq=selected.size(2),rows=groups*kv_blocks;
  TORCH_CHECK(groups>0 && groups<=65535 && sq>0 && selected.numel()<INT_MAX && rows>0 && rows<INT_MAX-1,"metadata dimensions");
  TORCH_CHECK(kv_blocks>=16 && kv_blocks<=4096,"initial metadata limit: 16..4096 KV blocks");
  TORCH_CHECK(g==1 || g==2 || g==4 || g==8 || g==16,"GQA ratio");
  TORCH_CHECK(chunk_size==1 || chunk_size==4 || chunk_size==8 || chunk_size==16,"chunk capacity");
  for(const auto& t:{counts,row_ptr,chunks,task_ptr})TORCH_CHECK(t.dim()==1 && t.numel()==rows+1,"scan buffers");
  TORCH_CHECK(cursors.dim()==1 && cursors.numel()==rows,"cursors");
  TORCH_CHECK(qids.dim()==1 && qids.numel()==selected.numel() && ranks.sizes()==qids.sizes(),"edge storage");
  TORCH_CHECK(error.numel()==1 && temp.dim()==1,"status/scratch");
  const int64_t capacity=128/g*chunk_size;
  const int64_t bound=std::min(selected.numel(),(selected.numel()+capacity-1)/capacity+rows);
  TORCH_CHECK(tasks.dim()==2 && tasks.size(1)==4 && tasks.size(0)==bound && bound<INT_MAX,"bounded tasks");
  const c10::cuda::CUDAGuard guard(selected.device());
  TORCH_CHECK(temp.numel()>=kv4_metadata_temp_bytes(rows),"CUB scratch too small");
  kv4_build_metadata(selected.data_ptr<int32_t>(),counts.data_ptr<int32_t>(),row_ptr.data_ptr<int32_t>(),
      cursors.data_ptr<int32_t>(),qids.data_ptr<int32_t>(),ranks.data_ptr<int32_t>(),chunks.data_ptr<int32_t>(),
      task_ptr.data_ptr<int32_t>(),tasks.data_ptr<int32_t>(),error.data_ptr<int32_t>(),temp.data_ptr(),temp.numel(),
      groups,sq,kv_blocks,g,chunk_size,bound,at::cuda::getCurrentCUDAStream().stream());
}
void main_out(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& tasks,const at::Tensor& qids,const at::Tensor& ranks,
    const at::Tensor& partial,const at::Tensor& lses,int64_t variant) {
  TORCH_CHECK(variant>=-2 && variant<=-1,"main_out serves the round9 controls; round11 variants use main_tm_out");
  for(const auto& t:{q,k,v})check(t,q,at::kBFloat16);
  for(const auto& t:{tasks,qids,ranks})check(t,q,at::kInt);
  const bool bf16=(variant==-1);
  check(partial,q,bf16?at::kBFloat16:at::kFloat);check(lses,q,at::kFloat);
  TORCH_CHECK(q.dim()==4 && k.dim()==4 && k.sizes()==v.sizes(),"QKV rank/shape");
  const auto b=q.size(0),hq=q.size(1),sq=q.size(2),hkv=k.size(1),skv=k.size(2);
  TORCH_CHECK(b>0 && sq>0 && hq>0 && hkv>0 && hq%hkv==0,"positive QKV/GQA shape");
  const auto g=hq/hkv;
  TORCH_CHECK(g==1 || g==2 || g==4 || g==8 || g==16,"GQA ratio");
  TORCH_CHECK(k.size(0)==b && q.size(3)==128 && k.size(3)==128 && skv>=2048 && skv%128==0,"D/block/batch");
  TORCH_CHECK(b*hq*sq<INT_MAX && b*hkv*skv<INT_MAX && b*hkv*sq*16<INT_MAX,"int32 limits");
  TORCH_CHECK(tasks.dim()==2 && tasks.size(1)==4 && tasks.size(0)>0 && tasks.size(0)<INT_MAX,"tasks");
  TORCH_CHECK(qids.dim()==1 && ranks.sizes()==qids.sizes() && qids.numel()==b*hkv*sq*16,"edges");
  TORCH_CHECK(partial.sizes()==at::IntArrayRef({b,hq,sq,16,128}) && lses.sizes()==at::IntArrayRef({b,hq,sq,16}),"partials");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto* prop=at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major==10 && prop->minor==0,"SM100 required");
  const auto* qp=reinterpret_cast<const nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  const auto* kp=reinterpret_cast<const nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  const auto* vp=reinterpret_cast<const nv_bfloat16*>(v.data_ptr<at::BFloat16>());
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  auto* tp=tasks.data_ptr<int32_t>();auto* ip=qids.data_ptr<int32_t>();auto* rp=ranks.data_ptr<int32_t>();
  auto* lp=lses.data_ptr<float>();
  kv9_launch(qp,kp,vp,tp,ip,rp,partial.data_ptr(),lp,b,hq,hkv,sq,skv,tasks.size(0),qids.numel(),variant+2,stream);
}
// Round11 entry: token-major Q and partial O.
void main_tm_out(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& tasks,const at::Tensor& qids,const at::Tensor& ranks,
    const at::Tensor& partial,const at::Tensor& lses,int64_t variant) {
  TORCH_CHECK(variant>=0 && variant<=2,"round11 variant");
  for(const auto& t:{q,k,v})check(t,q,at::kBFloat16);
  for(const auto& t:{tasks,qids,ranks})check(t,q,at::kInt);
  const bool bf16=(variant>=1);
  check(partial,q,bf16?at::kBFloat16:at::kFloat);check(lses,q,at::kFloat);
  TORCH_CHECK(q.dim()==4 && k.dim()==4 && k.sizes()==v.sizes(),"QKV rank/shape");
  const auto b=q.size(0),sq=q.size(1),hq=q.size(2),hkv=k.size(1),skv=k.size(2);
  TORCH_CHECK(b>0 && sq>0 && hq>0 && hkv>0 && hq%hkv==0,"positive QKV/GQA shape");
  const auto g=hq/hkv;
  TORCH_CHECK(g==1 || g==2 || g==4 || g==8 || g==16,"GQA ratio");
  TORCH_CHECK(k.size(0)==b && q.size(3)==128 && k.size(3)==128 && skv>=2048 && skv%128==0,"D/block/batch");
  TORCH_CHECK(b*hq*sq<INT_MAX && b*hkv*skv<INT_MAX && b*hkv*sq*16<INT_MAX && b*sq*hq*16<INT_MAX,"int32 limits");
  TORCH_CHECK(tasks.dim()==2 && tasks.size(1)==4 && tasks.size(0)>0 && tasks.size(0)<INT_MAX,"tasks");
  TORCH_CHECK(qids.dim()==1 && ranks.sizes()==qids.sizes() && qids.numel()==b*hkv*sq*16,"edges");
  TORCH_CHECK(partial.sizes()==at::IntArrayRef({b,sq,16,hq,128}) && lses.sizes()==at::IntArrayRef({b,hq,sq,16}),"token-major partials");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto* prop=at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major==10 && prop->minor==0,"SM100 required");
  const auto* qp=reinterpret_cast<const nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  const auto* kp=reinterpret_cast<const nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  const auto* vp=reinterpret_cast<const nv_bfloat16*>(v.data_ptr<at::BFloat16>());
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  kv11_launch(qp,kp,vp,tasks.data_ptr<int32_t>(),qids.data_ptr<int32_t>(),ranks.data_ptr<int32_t>(),partial.data_ptr(),
              lses.data_ptr<float>(),b,hq,hkv,sq,skv,tasks.size(0),qids.numel(),variant,stream);
}
void combine_tm_out(const at::Tensor& partial,const at::Tensor& lses,
    const at::Tensor& out,const at::Tensor& out_lse) {
  TORCH_CHECK(partial.scalar_type()==at::kFloat || partial.scalar_type()==at::kBFloat16,"partial dtype");
  check(partial,partial,partial.scalar_type());check(lses,partial,at::kFloat);
  check(out,partial,at::kBFloat16);check(out_lse,partial,at::kFloat);
  TORCH_CHECK(out.dim()==4 && out.size(3)==128 && out_lse.sizes()==out.sizes().slice(0,3),"output");
  const auto b=out.size(0),h=out.size(1),s=out.size(2);
  TORCH_CHECK(b>0 && h>0 && s>0 && out_lse.numel()<INT_MAX,"output limits");
  TORCH_CHECK(partial.sizes()==at::IntArrayRef({b,s,16,h,128}) && lses.sizes()==at::IntArrayRef({b,h,s,16}),"token-major Combine input");
  const c10::cuda::CUDAGuard guard(out.device());
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  auto* op=reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  if(partial.scalar_type()==at::kFloat)
    kv11_launch_combine_f32(partial.data_ptr<float>(),lses.data_ptr<float>(),op,out_lse.data_ptr<float>(),out_lse.numel(),h,s,stream);
  else
    kv11_launch_combine_bf16(reinterpret_cast<const nv_bfloat16*>(partial.data_ptr<at::BFloat16>()),lses.data_ptr<float>(),
                             op,out_lse.data_ptr<float>(),out_lse.numel(),h,s,stream);
}
void combine_out(const at::Tensor& partial,const at::Tensor& lses,
    const at::Tensor& out,const at::Tensor& out_lse) {
  TORCH_CHECK(partial.scalar_type()==at::kFloat || partial.scalar_type()==at::kBFloat16,"partial dtype");
  check(partial,partial,partial.scalar_type());check(lses,partial,at::kFloat);
  check(out,partial,at::kBFloat16);check(out_lse,partial,at::kFloat);
  TORCH_CHECK(out.dim()==4 && out.size(3)==128 && out_lse.sizes()==out.sizes().slice(0,3),"output");
  const auto b=out.size(0),h=out.size(1),s=out.size(2);
  TORCH_CHECK(b>0 && h>0 && s>0 && out_lse.numel()<INT_MAX,"output limits");
  TORCH_CHECK(partial.sizes()==at::IntArrayRef({b,h,s,16,128}) && lses.sizes()==at::IntArrayRef({b,h,s,16}),"Combine input");
  const c10::cuda::CUDAGuard guard(out.device());
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  auto* op=reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  if(partial.scalar_type()==at::kFloat)
    kv1_launch_combine(partial.data_ptr<float>(),lses.data_ptr<float>(),op,out_lse.data_ptr<float>(),out_lse.numel(),stream);
  else
    kv8_launch_combine_bf16(reinterpret_cast<const nv_bfloat16*>(partial.data_ptr<at::BFloat16>()),lses.data_ptr<float>(),
                            op,out_lse.data_ptr<float>(),out_lse.numel(),stream);
}
}
TORCH_LIBRARY(xomtention_kvopt11,m) {
  m.def("metadata_temp_bytes(int rows) -> int",&metadata_temp_bytes);
  m.def("metadata_out(Tensor selected, Tensor(a!) counts, Tensor(b!) row_ptr, Tensor(c!) cursors, Tensor(d!) qids, Tensor(e!) ranks, Tensor(f!) chunks, Tensor(g!) task_ptr, Tensor(h!) tasks, Tensor(i!) error, Tensor(j!) temp, int kv_blocks, int group_size, int chunk_size) -> ()");
  m.def("main_out(Tensor q, Tensor k, Tensor v, Tensor tasks, Tensor qids, Tensor ranks, Tensor(a!) partial, Tensor(b!) lses, int variant) -> ()");
  m.def("combine_out(Tensor partial, Tensor lses, Tensor(a!) out, Tensor(b!) out_lse) -> ()");
  m.def("main_tm_out(Tensor q, Tensor k, Tensor v, Tensor tasks, Tensor qids, Tensor ranks, Tensor(a!) partial, Tensor(b!) lses, int variant) -> ()");
  m.def("combine_tm_out(Tensor partial, Tensor lses, Tensor(a!) out, Tensor(b!) out_lse) -> ()");
}
TORCH_LIBRARY_IMPL(xomtention_kvopt11,CUDA,m) {
  m.impl("metadata_out",&metadata_out);
  m.impl("main_out",&main_out);m.impl("combine_out",&combine_out);
  m.impl("main_tm_out",&main_tm_out);m.impl("combine_tm_out",&combine_tm_out);
}
