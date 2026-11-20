#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ext_prefill.h"
#include "cuda/prefill_hd512.cuh"

void prefill_hd512
(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    double sm_scale
)
{
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kHalf, "prefill_hd512: q must be CUDA fp16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && o.is_contiguous(),
                "prefill_hd512: tensors must be contiguous");
    TORCH_CHECK(kv_len >= q_len && q_len > 0, "prefill_hd512: requires kv_len >= q_len > 0");
    TORCH_CHECK(q.size(2) == 512, "prefill_hd512: head_dim must be 512");

    const c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    prefill_hd512_cuda
    (
        (const half*) q.data_ptr(),
        (const half*) k.data_ptr(),
        (const half*) v.data_ptr(),
        (half*) o.data_ptr(),
        (int) q_len,
        (int) kv_len,
        (int) num_qo_heads,
        (int) num_kv_heads,
        (float) sm_scale,
        stream
    );
}