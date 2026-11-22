import sys, os, torch
from torch.utils.cpp_extension import load_inline

ROOT = os.path.join(os.path.dirname(__file__), "..", "exllamav2", "exllamav2_ext")
cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda/exl25_gemm.cuh"

void dbg_rw(torch::Tensor a, torch::Tensor qw, torch::Tensor qs, torch::Tensor qsm,
            torch::Tensor c, torch::Tensor rw, int gs, int mulr, int bits)
{
    int m = a.size(0), k = a.size(1), n = c.size(1);
    dim3 block(128), grid((n + 511) / 512, m, k / 128);
    auto s = at::cuda::getCurrentCUDAStream();
    if (bits == 6) {
        if (mulr) exl25_gemm_kernel<6, 1, 128, true, true><<<grid, block, 0, s>>>(
            (const half*)a.data_ptr(), (const uint32_t*)qw.data_ptr(), (const uint32_t*)qs.data_ptr(),
            (const half*)qsm.data_ptr(), (half*)c.data_ptr(), m, n, k, gs,
            nullptr, 0, true, (const half*)rw.data_ptr(), 8);
        else exl25_gemm_kernel<6, 1, 128, true, false><<<grid, block, 0, s>>>(
            (const half*)a.data_ptr(), (const uint32_t*)qw.data_ptr(), (const uint32_t*)qs.data_ptr(),
            (const half*)qsm.data_ptr(), (half*)c.data_ptr(), m, n, k, gs,
            nullptr, 0, true, (const half*)rw.data_ptr(), 8);
    } else {
        if (mulr) exl25_gemm_kernel<4, 1, 128, true, true><<<grid, block, 0, s>>>(
            (const half*)a.data_ptr(), (const uint32_t*)qw.data_ptr(), (const uint32_t*)qs.data_ptr(),
            (const half*)qsm.data_ptr(), (half*)c.data_ptr(), m, n, k, gs,
            nullptr, 0, true, (const half*)rw.data_ptr(), 8);
        else exl25_gemm_kernel<4, 1, 128, true, false><<<grid, block, 0, s>>>(
            (const half*)a.data_ptr(), (const uint32_t*)qw.data_ptr(), (const uint32_t*)qs.data_ptr(),
            (const half*)qsm.data_ptr(), (half*)c.data_ptr(), m, n, k, gs,
            nullptr, 0, true, (const half*)rw.data_ptr(), 8);
    }
}
'''
cpp_src = "void dbg_rw(torch::Tensor a, torch::Tensor qw, torch::Tensor qs, torch::Tensor qsm, torch::Tensor c, torch::Tensor rw, int gs, int mulr, int bits);"
mod = load_inline(name="exl25_dbg_rw", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  extra_include_paths=[ROOT], extra_cuda_cflags=["-O3"], functions=["dbg_rw"], verbose=False)

DEV = "cuda:0"
k, n, gs = 1024, 512, 128
k4rows = k * 4 // 32
# uniform q=9 nibbles (shuffle-invariant) -> dq = 1 * (15+1)^2 * qsm = 256*qsm
qw4 = torch.full((k4rows, n), 0x99999999 - (1 << 32), dtype=torch.int32, device=DEV)
qs = torch.full((k // gs, n // 8), -1, dtype=torch.int32, device=DEV)
qsm = torch.full((k // gs,), 0.04, dtype=torch.half, device=DEV)  # JIT: no /256 fold
x = torch.ones(1, k, dtype=torch.half, device=DEV)
rw = torch.tensor([0.5] + [0.0] * 7, dtype=torch.half, device=DEV)
for mulr in (0, 1):
    c = torch.zeros(1, n, dtype=torch.half, device=DEV)
    mod.dbg_rw(x, qw4, qs, qsm, c, rw, gs, mulr, 4)
    torch.cuda.synchronize()
    expect = (k * 256 * 0.04) * (0.5 if mulr else 1.0)
    print(f"4b mulr={mulr}: c[0,0]={c[0,0].item():.4f} expect {expect:.4f}")
# zero weight -> early exit, c must stay 0
rw0 = torch.zeros(8, dtype=torch.half, device=DEV)
c = torch.full((1, n), 7.0, dtype=torch.half, device=DEV)
mod.dbg_rw(x, qw4, qs, qsm, c, rw0, gs, 1, 4)
torch.cuda.synchronize()
print(f"zero-weight early exit: c[0,0]={c[0,0].item():.4f} expect 7.0 (untouched)")
