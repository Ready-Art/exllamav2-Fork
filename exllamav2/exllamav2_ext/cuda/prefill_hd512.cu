#include "prefill_hd512.cuh"
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define HD5 512
#define BM5 16
#define BN5 64
#define NW5 4

// Shared memory: Q tile (16x512), one KV tile (64x512, K then V reusing the buffer),
// four 16x16 P tiles, and the cross-warp row-max/row-sum scratch (4 warps x 16 rows x 2).
#define SMEM_BYTES5 ((BM5*HD5 + BN5*HD5 + NW5*BM5*BM5)*2 + NW5*BM5*2*4)

// wmma 16x16x16 fp32 accumulator element map (sm80/sm86, matches mma.m16n8k16):
//   e0,e1 -> row g,   cols 2t, 2t+1      e2,e3 -> row g+8, cols 2t, 2t+1
//   e4,e5 -> row g,   cols 2t+8, 2t+9    e6,e7 -> row g+8, cols 2t+8, 2t+9
// with g = lane/4, t = lane%4.

__global__ void __launch_bounds__(128, 1)
prefill_hd512_kernel
(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ o,
    int q_len,
    int kv_len,
    int num_qo_heads,
    int num_kv_heads,
    float scale_log2e
)
{
    extern __shared__ char smem_raw[];
    half* sQ = (half*)smem_raw;
    half* sKV = sQ + BM5*HD5;
    half* sP = sKV + BN5*HD5;
    float* sRed = (float*)(sP + NW5*BM5*BM5);

    const int q0 = blockIdx.x * BM5;
    const int h = blockIdx.y;
    const int kvh = h / (num_qo_heads / num_kv_heads);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const long stride_q = (long)num_qo_heads * HD5;
    const long stride_kv = (long)num_kv_heads * HD5;

    // Q tile, pre-scaled by sm_scale * log2(e) so the softmax can use exp2f.
    for (int i = tid; i < BM5*HD5/8; i += 128)
    {
        int r = i / (HD5/8);
        int c = (i % (HD5/8)) * 8;
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q0 + r < q_len)
        {
            val = *(const float4*)(q + (long)(q0 + r)*stride_q + (long)h*HD5 + c);
            half* hv = (half*)&val;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                hv[j] = __float2half(__half2float(hv[j]) * scale_log2e);
        }
        *(float4*)(sQ + r*HD5 + c) = val;
    }
    __syncthreads();

    // Warp w owns output dims [w*128, w*128+128) -> 8 accumulator tiles.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(acc[j], 0.0f);
    float m_i[2] = {-1e30f, -1e30f};
    float l_i[2] = {0.0f, 0.0f};

    // Lower-right causal: row (q0+r) attends keys <= kv_len - q_len + q0 + r.
    int hi = kv_len - q_len + q0 + BM5;
    if (hi > kv_len) hi = kv_len;
    if (hi < 0) hi = 0;

    for (int n0 = 0; n0 < hi; n0 += BN5)
    {
        // ---- K tile (64 keys x 512 dims) ----
        for (int i = tid; i < BN5*HD5/8; i += 128)
        {
            int r = i / (HD5/8);
            int c = (i % (HD5/8)) * 8;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (n0 + r < kv_len)
                val = *(const float4*)(k + (long)(n0 + r)*stride_kv + (long)kvh*HD5 + c);
            *(float4*)(sKV + r*HD5 + c) = val;
        }
        __syncthreads();

        // ---- QK^T: warp w owns key cols [w*16, w*16+16) ----
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
        wmma::fill_fragment(sc, 0.0f);
        #pragma unroll
        for (int kc = 0; kc < HD5/16; kc++)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
            wmma::load_matrix_sync(qa, sQ + kc*16, HD5);
            // col_major B = K^T: element (dim, key) at sKV[key*512 + dim]
            wmma::load_matrix_sync(kb, sKV + (warp*16)*HD5 + kc*16, HD5);
            wmma::mma_sync(sc, qa, kb, sc);
        }

        // causal mask + per-row max (rows g and g+8 within this warp's key tile)
        float mx[2] = {-1e30f, -1e30f};
        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int row = g + ((e & 2) ? 8 : 0);
            int col = 2*t + ((e & 4) ? 8 : 0) + (e & 1);
            int key = n0 + warp*16 + col;
            bool ok = (q0 + row < q_len) && (key < kv_len) &&
                      (key <= kv_len - q_len + q0 + row);
            if (!ok) sc.x[e] = -1e30f;
            int ri = (e & 2) ? 1 : 0;
            mx[ri] = fmaxf(mx[ri], sc.x[e]);
        }
        mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffffu, mx[0], 1));
        mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffffu, mx[0], 2));
        mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffffu, mx[1], 1));
        mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffffu, mx[1], 2));
        if (t == 0) { sRed[warp*16 + g] = mx[0]; sRed[warp*16 + g + 8] = mx[1]; }
        __syncthreads();
        float mnew[2];
        mnew[0] = fmaxf(m_i[0], fmaxf(fmaxf(sRed[g], sRed[16 + g]), fmaxf(sRed[32 + g], sRed[48 + g])));
        mnew[1] = fmaxf(m_i[1], fmaxf(fmaxf(sRed[g + 8], sRed[16 + g + 8]), fmaxf(sRed[32 + g + 8], sRed[48 + g + 8])));
        __syncthreads();  // sRed reused for row sums below

        float alpha[2] = { exp2f(m_i[0] - mnew[0]), exp2f(m_i[1] - mnew[1]) };
        float ps[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int ri = (e & 2) ? 1 : 0;
            sc.x[e] = exp2f(sc.x[e] - mnew[ri]);
            ps[ri] += sc.x[e];
        }
        ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], 1);
        ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], 2);
        ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], 1);
        ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], 2);
        if (t == 0) { sRed[warp*16 + g] = ps[0]; sRed[warp*16 + g + 8] = ps[1]; }
        __syncthreads();
        float lnew[2];
        lnew[0] = l_i[0]*alpha[0] + sRed[g] + sRed[16 + g] + sRed[32 + g] + sRed[48 + g];
        lnew[1] = l_i[1]*alpha[1] + sRed[g + 8] + sRed[16 + g + 8] + sRed[32 + g + 8] + sRed[48 + g + 8];

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            #pragma unroll
            for (int e = 0; e < 8; e++)
                acc[j].x[e] *= alpha[(e & 2) ? 1 : 0];
        }

        // share the four P tiles (fp16) for the PV phase
        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int row = g + ((e & 2) ? 8 : 0);
            int col = 2*t + ((e & 4) ? 8 : 0) + (e & 1);
            sP[warp*256 + row*16 + col] = __float2half(sc.x[e]);
        }
        __syncthreads();

        // ---- V tile reuses the KV buffer ----
        for (int i = tid; i < BN5*HD5/8; i += 128)
        {
            int r = i / (HD5/8);
            int c = (i % (HD5/8)) * 8;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (n0 + r < kv_len)
                val = *(const float4*)(v + (long)(n0 + r)*stride_kv + (long)kvh*HD5 + c);
            *(float4*)(sKV + r*HD5 + c) = val;
        }
        __syncthreads();

        // ---- PV: acc[j] += sum_t P_t * V[keys t*16.., dims w*128+j*16..] ----
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            #pragma unroll
            for (int tt = 0; tt < 4; tt++)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
                wmma::load_matrix_sync(pa, sP + tt*256, 16);
                wmma::load_matrix_sync(vb, sKV + (tt*16)*HD5 + warp*128 + j*16, HD5);
                wmma::mma_sync(acc[j], pa, vb, acc[j]);
            }
        }

        m_i[0] = mnew[0]; m_i[1] = mnew[1];
        l_i[0] = lnew[0]; l_i[1] = lnew[1];
        __syncthreads();  // before the next K tile overwrites sKV
    }

    // ---- epilogue: normalize and store ----
    float inv[2] = { (l_i[0] > 0.0f) ? 1.0f/l_i[0] : 0.0f,
                     (l_i[1] > 0.0f) ? 1.0f/l_i[1] : 0.0f };
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int row = g + ((e & 2) ? 8 : 0);
            int col = 2*t + ((e & 4) ? 8 : 0) + (e & 1);
            if (q0 + row < q_len)
                o[(long)(q0 + row)*stride_q + (long)h*HD5 + warp*128 + j*16 + col] =
                    __float2half(acc[j].x[e] * inv[(e & 2) ? 1 : 0]);
        }
    }
}

void prefill_hd512_cuda
(
    const half* q,
    const half* k,
    const half* v,
    half* o,
    int q_len,
    int kv_len,
    int num_qo_heads,
    int num_kv_heads,
    float sm_scale,
    cudaStream_t stream
)
{
    // Per-call: the attribute is per-device, and single-process TP spans devices.
    cudaFuncSetAttribute(prefill_hd512_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES5);
    dim3 grid((q_len + BM5 - 1) / BM5, num_qo_heads);
    float scale_log2e = sm_scale * 1.4426950408889634f;
    prefill_hd512_kernel<<<grid, 128, SMEM_BYTES5, stream>>>
        (q, k, v, o, q_len, kv_len, num_qo_heads, num_kv_heads, scale_log2e);
}