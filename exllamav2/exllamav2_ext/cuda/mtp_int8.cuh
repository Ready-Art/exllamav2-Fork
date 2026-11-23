#ifndef _MTP_INT8_CUH
#define _MTP_INT8_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Weight-only INT8 GEMV for the MTP drafter (Ampere, no FP8):
// out(m, N) = x(m, K) fp16 @ w8(N, K)^T * scale(N) fp16, m <= 8.
void mtp_int8_gemm_cuda
(
    const half* x,
    const int8_t* w8,
    const half* scale,
    half* out,
    int m,
    int N,
    int K,
    cudaStream_t stream
);

#endif
