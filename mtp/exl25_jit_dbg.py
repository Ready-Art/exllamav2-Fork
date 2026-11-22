import sys, os, torch
from torch.utils.cpp_extension import load_inline

ROOT = os.path.join(os.path.dirname(__file__), "..", "exllamav2", "exllamav2_ext")
cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda/exl25_gemm.cuh"

void dbg_gemm(torch::Tensor a, torch::Tensor qw, torch::Tensor qs, torch::Tensor qsm, torch::Tensor c, int gs)
{
    int m = a.size(0), k = a.size(1), n = c.size(1);
    dim3 block(128), grid((n + 511) / 512, m, k / 128);
    exl25_gemm_kernel<4, 1, 128, false, false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)a.data_ptr(), (const uint32_t*)qw.data_ptr(), (const uint32_t*)qs.data_ptr(),
        (const half*)qsm.data_ptr(), (half*)c.data_ptr(), m, n, k, gs, nullptr, 0, true, nullptr, 0);
}
'''
cpp_src = "void dbg_gemm(torch::Tensor a, torch::Tensor qw, torch::Tensor qs, torch::Tensor qsm, torch::Tensor c, int gs);"
mod = load_inline(name="exl25_dbg", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  extra_include_paths=[ROOT], extra_cuda_cflags=["-O3", "-lineinfo"],
                  functions=["dbg_gemm"], verbose=False)

DEV = "cuda:0"
k, n, gs = 256, 512, 128
qrows = k * 4 // 32
qw = torch.full((qrows, n), 0x99999999 - (1 << 32), dtype=torch.int32, device=DEV)
qs = torch.full((k // gs, n // 8), -1, dtype=torch.int32, device=DEV)
qsm = torch.tensor([256 * 0.04, 256 * 0.08], dtype=torch.half, device=DEV)
# NOTE: no shuffle applied here (JIT bypasses QMatrix) -- all-equal nibbles are shuffle-invariant.
x = torch.zeros(1, k, dtype=torch.half, device=DEV); x[0, 5] = 1.0
c = torch.zeros(1, n, dtype=torch.half, device=DEV)
mod.dbg_gemm(x, qw, qs, qsm, c, gs)
torch.cuda.synchronize()
print("one-hot j=5 -> c[0,0] (expect 0.04):", c[0, 0].item())
