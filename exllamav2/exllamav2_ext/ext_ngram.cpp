#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "config.h"
#include "ext_ngram.h"
#include "cuda/ngram.cuh"
#include "cpp/util.h"

void ngram_dequant_rows
(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor heads,
    torch::Tensor bias,
    torch::Tensor out,
    int bits
)
{
    TORCH_CHECK_DTYPE(packed, kInt);
    TORCH_CHECK_DTYPE(scales, kHalf);
    TORCH_CHECK_DTYPE(heads, kInt);
    TORCH_CHECK_DTYPE(bias, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(bits == 4 || bits == 8, "ngram bits must be 4 or 8");

    int num_rows = out.size(0);
    int dim = out.size(1);
    TORCH_CHECK(dim % (32 / bits) == 0, "ngram dim must divide the packed word");
    TORCH_CHECK_SHAPES(packed, 0, out, 0, 1);
    TORCH_CHECK(packed.size(1) == dim * bits / 32, "ngram packed row width mismatch");
    TORCH_CHECK_SHAPES(scales, 0, out, 0, 1);
    TORCH_CHECK_SHAPES(heads, 0, out, 0, 1);
    TORCH_CHECK(bias.size(1) == dim, "ngram bias dim mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ngram_dequant_rows_cuda
    (
        stream,
        (const uint32_t*) packed.data_ptr(),
        (const half*) scales.data_ptr(),
        (const int32_t*) heads.data_ptr(),
        (const half*) bias.data_ptr(),
        (half*) out.data_ptr(),
        num_rows,
        dim,
        bits
    );
}
