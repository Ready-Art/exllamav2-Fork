#include "mtp_int8.cuh"

// Weight-only INT8 GEMV: out(m, N) = x(m, K) fp16 @ w8(N, K)^T * scale(N) fp16.
// One WARP per output row (8 rows per 256-thread CTA): lanes stripe K with char4
// loads, fp32 accumulate, shfl reduction (no shared memory / block syncs). The
// weight matrix streams exactly once; x (a few KB) stays L2-resident across rows
// and activation rows. Bandwidth-bound by design (drafter m=1..B GEMVs).

template <int M>
__global__ void __launch_bounds__(256)
mtp_int8_gemm_kernel
(
    const half* __restrict__ x,
    const int8_t* __restrict__ w8,
    const half* __restrict__ scale,
    half* __restrict__ out,
    int N,
    int K
)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;

    const int8_t* wrow = w8 + (long)n * K;
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    const int K4 = K >> 2;
    for (int g = lane; g < K4; g += 32)
    {
        const int k = g << 2;
        const int wv = *(const int*)(wrow + k);
        const float w0 = (float)(int8_t)(wv & 0xff);
        const float w1 = (float)(int8_t)((wv >> 8) & 0xff);
        const float w2 = (float)(int8_t)((wv >> 16) & 0xff);
        const float w3 = (float)(int8_t)((wv >> 24) & 0xff);
        #pragma unroll
        for (int m = 0; m < M; m++)
        {
            const half* xr = x + (long)m * K + k;
            acc[m] = fmaf(__half2float(xr[0]), w0, acc[m]);
            acc[m] = fmaf(__half2float(xr[1]), w1, acc[m]);
            acc[m] = fmaf(__half2float(xr[2]), w2, acc[m]);
            acc[m] = fmaf(__half2float(xr[3]), w3, acc[m]);
        }
    }
    for (int k = (K4 << 2) + lane; k < K; k += 32)
    {
        const float wv = (float)wrow[k];
        #pragma unroll
        for (int m = 0; m < M; m++)
            acc[m] = fmaf(__half2float(x[(long)m * K + k]), wv, acc[m]);
    }

    #pragma unroll
    for (int m = 0; m < M; m++)
    {
        float s = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0)
            out[(long)m * N + n] = __float2half(s * __half2float(scale[n]));
    }
}

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
)
{
    dim3 grid((N + 7) / 8);
    switch (m)
    {
        case 1: mtp_int8_gemm_kernel<1><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 2: mtp_int8_gemm_kernel<2><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 3: mtp_int8_gemm_kernel<3><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 4: mtp_int8_gemm_kernel<4><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 5: mtp_int8_gemm_kernel<5><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 6: mtp_int8_gemm_kernel<6><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 7: mtp_int8_gemm_kernel<7><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        case 8: mtp_int8_gemm_kernel<8><<<grid, 256, 0, stream>>>(x, w8, scale, out, N, K); break;
        default: break;
    }
}
