#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ext_verify.h"
#include "cuda/verify_hd512.cuh"

void verify_hd512
(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int64_t q_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t num_splits,
    double sm_scale
)
{
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kHalf, "verify_hd512: q must be CUDA fp16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && o.is_contiguous(),
                "verify_hd512: q/k/v/o must be contiguous");
    TORCH_CHECK(po.is_contiguous() && pl.is_contiguous() && po.dtype() == torch::kFloat
                && pl.dtype() == torch::kFloat, "verify_hd512: po/pl must be contiguous fp32");
    TORCH_CHECK(kv_len_dev.is_cuda() && kv_len_dev.dtype() == torch::kInt
                && kv_len_dev.numel() == 1, "verify_hd512: kv_len_dev must be a CUDA int32 scalar");
    TORCH_CHECK(q_len > 0, "verify_hd512: requires q_len > 0");
    TORCH_CHECK(q.size(2) == 512, "verify_hd512: head_dim must be 512");
    TORCH_CHECK(num_splits > 0, "verify_hd512: num_splits must be > 0");
    TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "verify_hd512: GQA group must divide");

    const c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    verify_hd512_cuda
    (
        (const half*) q.data_ptr(),
        (const half*) k.data_ptr(),
        (const half*) v.data_ptr(),
        (half*) o.data_ptr(),
        (float*) po.data_ptr(),
        (float*) pl.data_ptr(),
        (int) q_len,
        (const int*) kv_len_dev.data_ptr(),
        (int) num_qo_heads,
        (int) num_kv_heads,
        (int) num_splits,
        (float) sm_scale,
        stream
    );
}