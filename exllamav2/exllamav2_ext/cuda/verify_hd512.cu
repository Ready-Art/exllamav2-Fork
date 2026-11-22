#include "verify_hd512.cuh"
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define HDV 512
#define BMV 16
#define BNV 64
#define NWV 4

// Shared memory: Q tile (16x512), one KV tile (64x512, K then V reusing the buffer),
// four 16x16 P tiles, and the cross-warp row-max/row-sum scratch (4 warps x 16 rows x 2).
#define SMEM_BYTESV ((BMV*HDV + BNV*HDV + NWV*BMV*BMV)*2 + NWV*BMV*2*4)

// wmma 16x16x16 fp32 accumulator element map (sm80/sm86, matches mma.m16n8k16):
//   e0,e1 -> row g,   cols 2t, 2t+1      e2,e3 -> row g+8, cols 2t, 2t+1
//   e4,e5 -> row g,   cols 2t+8, 2t+9    e6,e7 -> row g+8, cols 2t+8, 2t+9
// with g = lane/4, t = lane%4.

// Partial attention over one KV chunk [kv_begin, kv_end) for one query tile (q0) and
// one q-head (h). Writes UNNORMALIZED online-softmax partials (po = sum exp2(s-m)*V,
// pl = (m, l)) so the merge kernel can combine splits with a global log-sum-exp.
//
// GRID ORDERING IS LOAD-BEARING: blockIdx.x = q-head (fastest), blockIdx.y = KV split.
// The GQA group's heads share one KV head, so the co-head CTAs of a split must be
// co-resident to reuse that split's KV chunk through L2 (read once from HBM, not once
// per head). With heads in the slow grid dim they were scheduled a full split apart and
// every head re-read the chunk from HBM (~group x traffic) -- the long-ctx regression.
__global__ void __launch_bounds__(128, 1)
verify_hd512_partial_kernel
(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ po,
    float* __restrict__ pl,
    int q_len,
    const int* __restrict__ kv_len_dev,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float scale_log2e
)
{
    const int kv_len = *kv_len_dev;
    extern __shared__ char smem_raw[];
    half* sQ = (half*)smem_raw;
    half* sKV = sQ + BMV*HDV;
    half* sP = sKV + BNV*HDV;
    float* sRed = (float*)(sP + NWV*BMV*BMV);

    const int h = blockIdx.x;
    const int split = blockIdx.y;
    const int kvh = h / (num_qo_heads / num_kv_heads);
    const int q0 = blockIdx.z * BMV;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const long stride_q = (long)num_qo_heads * HDV;
    const long stride_kv = (long)num_kv_heads * HDV;

    // This split's KV chunk. The last split absorbs the remainder.
    const int chunk = (kv_len + num_splits - 1) / num_splits;
    const int kv_begin = split * chunk;
    int kv_end = kv_begin + chunk;
    if (kv_end > kv_len) kv_end = kv_len;

    // Q tile, pre-scaled by sm_scale * log2(e) so the softmax can use exp2f.
    for (int i = tid; i < BMV*HDV/8; i += 128)
    {
        int r = i / (HDV/8);
        int c = (i % (HDV/8)) * 8;
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q0 + r < q_len)
        {
            val = *(const float4*)(q + (long)(q0 + r)*stride_q + (long)h*HDV + c);
            half* hv = (half*)&val;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                hv[j] = __float2half(__half2float(hv[j]) * scale_log2e);
        }
        *(float4*)(sQ + r*HDV + c) = val;
    }
    __syncthreads();

    // Warp w owns output dims [w*128, w*128+128) -> 8 accumulator tiles.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(acc[j], 0.0f);
    float m_i[2] = {-1e30f, -1e30f};
    float l_i[2] = {0.0f, 0.0f};

    // Lower-right causal: row (q0+r) attends keys <= kv_len - q_len + q0 + r.
    int hi = kv_len - q_len + q0 + BMV;
    if (hi > kv_end) hi = kv_end;
    if (hi < kv_begin) hi = kv_begin;

    for (int n0 = kv_begin; n0 < hi; n0 += BNV)
    {
        // ---- K tile (64 keys x 512 dims) ----
        for (int i = tid; i < BNV*HDV/8; i += 128)
        {
            int r = i / (HDV/8);
            int c = (i % (HDV/8)) * 8;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (n0 + r < kv_len)
                val = *(const float4*)(k + (long)(n0 + r)*stride_kv + (long)kvh*HDV + c);
            *(float4*)(sKV + r*HDV + c) = val;
        }
        __syncthreads();

        // ---- QK^T: warp w owns key cols [w*16, w*16+16) ----
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
        wmma::fill_fragment(sc, 0.0f);
        #pragma unroll
        for (int kc = 0; kc < HDV/16; kc++)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
            wmma::load_matrix_sync(qa, sQ + kc*16, HDV);
            wmma::load_matrix_sync(kb, sKV + (warp*16)*HDV + kc*16, HDV);
            wmma::mma_sync(sc, qa, kb, sc);
        }

        float mx[2] = {-1e30f, -1e30f};
        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int row = g + ((e & 2) ? 8 : 0);
            int col = 2*t + ((e & 4) ? 8 : 0) + (e & 1);
            int key = n0 + warp*16 + col;
            // key < kv_end is LOAD-BEARING: a split's last 64-key tile can overshoot its
            // chunk boundary (chunk need not be a multiple of 64); without this bound those
            // keys are also processed by the NEXT split -> double-counted in the merge
            // (the kv1315/79067 parity failures; kv4096/32768 passed only because their
            // chunk happened to be a multiple of 64).
            bool ok = (q0 + row < q_len) && (key < kv_end) &&
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
        __syncthreads();

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

        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int row = g + ((e & 2) ? 8 : 0);
            int col = 2*t + ((e & 4) ? 8 : 0) + (e & 1);
            sP[warp*256 + row*16 + col] = __float2half(sc.x[e]);
        }
        __syncthreads();

        // ---- V tile reuses the KV buffer ----
        for (int i = tid; i < BNV*HDV/8; i += 128)
        {
            int r = i / (HDV/8);
            int c = (i % (HDV/8)) * 8;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (n0 + r < kv_len)
                val = *(const float4*)(v + (long)(n0 + r)*stride_kv + (long)kvh*HDV + c);
            *(float4*)(sKV + r*HDV + c) = val;
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
                wmma::load_matrix_sync(vb, sKV + (tt*16)*HDV + warp*128 + j*16, HDV);
                wmma::mma_sync(acc[j], pa, vb, acc[j]);
            }
        }

        m_i[0] = mnew[0]; m_i[1] = mnew[1];
        l_i[0] = lnew[0]; l_i[1] = lnew[1];
        __syncthreads();
    }

    // ---- epilogue: write UNNORMALIZED partials (acc, m_i, l_i) for the merge ----
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        #pragma unroll
        for (int e = 0; e < 8; e++)
        {
            int row = g + ((e & 2) ? 8 : 0);
            int col = 2*t + ((e & 4) ? 8 : 0) + (e & 1);
            if (q0 + row < q_len)
            {
                long base = ((long)(split * q_len + (q0 + row)) * num_qo_heads + h) * HDV;
                po[base + warp*128 + j*16 + col] = acc[j].x[e];
            }
        }
    }
    // m_i/l_i are identical across warps (cross-warp sRed reduction); warp 0 writes.
    if (t == 0 && warp == 0)
    {
        #pragma unroll
        for (int ri = 0; ri < 2; ri++)
        {
            int row = g + ri*8;
            if (q0 + row < q_len)
            {
                long idx = ((long)(split * q_len + (q0 + row)) * num_qo_heads + h) * 2;
                pl[idx + 0] = m_i[ri];
                pl[idx + 1] = l_i[ri];
            }
        }
    }
}

// Merge the split partials with a global log-sum-exp. One BLOCK per (row, head); the
// 512 output dims are striped across the block's threads so every po/pl read is fully
// coalesced (the per-element-thread version read po with a 512-float stride -> ~32x
// wasted bandwidth). blockDim.x must be a multiple of 32 and divide 512.
__global__ void
verify_hd512_merge_kernel
(
    const float* __restrict__ po,
    const float* __restrict__ pl,
    half* __restrict__ o,
    int q_len,
    int num_qo_heads,
    int num_splits
)
{
    const long rh = blockIdx.x;             // (row * num_qo_heads + head)
    const int h = rh % num_qo_heads;
    const int r = rh / num_qo_heads;
    const int d = threadIdx.x;              // 0..511 (blockDim.x == 512)

    __shared__ float s_w[64];               // split weights (num_splits <= 64)
    __shared__ float s_gm;
    __shared__ float s_den;

    // global max over splits (thread 0; num_splits is small)
    if (d == 0)
    {
        float gm = -1e30f;
        for (int s = 0; s < num_splits; s++)
        {
            long mi = ((long)(s * q_len + r) * num_qo_heads + h) * 2;
            gm = fmaxf(gm, pl[mi + 0]);
        }
        s_gm = gm;
    }
    __syncthreads();
    const float gm = s_gm;

    // split weights w_s = exp2(m_s - gm), and the shared denominator sum w_s * l_s
    if (d < num_splits)
    {
        long mi = ((long)(d * q_len + r) * num_qo_heads + h) * 2;
        float w = exp2f(pl[mi + 0] - gm);
        s_w[d] = w;
        s_den = 0.0f;
    }
    __syncthreads();
    if (d < num_splits)
    {
        long mi = ((long)(d * q_len + r) * num_qo_heads + h) * 2;
        atomicAdd(&s_den, s_w[d] * pl[mi + 1]);
    }
    __syncthreads();
    const float inv = (s_den > 0.0f) ? 1.0f / s_den : 0.0f;

    // coalesced accumulate over splits for this thread's output dim d
    float num = 0.0f;
    for (int s = 0; s < num_splits; s++)
    {
        long pi = ((long)(s * q_len + r) * num_qo_heads + h) * HDV + d;
        num += s_w[s] * po[pi];
    }
    o[rh * HDV + d] = __float2half(num * inv);
}

void verify_hd512_cuda
(
    const half* q,
    const half* k,
    const half* v,
    half* o,
    float* po,
    float* pl,
    int q_len,
    const int* kv_len_dev,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float sm_scale,
    cudaStream_t stream
)
{
    cudaFuncSetAttribute(verify_hd512_partial_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTESV);
    float scale_log2e = sm_scale * 1.4426950408889634f;
    // heads fastest (blockIdx.x) so a GQA group's co-head CTAs of a split are co-resident
    // and reuse the split's KV chunk through L2; splits in blockIdx.y.
    dim3 grid(num_qo_heads, num_splits, (q_len + BMV - 1) / BMV);
    verify_hd512_partial_kernel<<<grid, 128, SMEM_BYTESV, stream>>>
        (q, k, v, po, pl, q_len, kv_len_dev, num_qo_heads, num_kv_heads, num_splits, scale_log2e);

    long blocks = (long)q_len * num_qo_heads;
    verify_hd512_merge_kernel<<<(int)blocks, HDV, 0, stream>>>
        (po, pl, o, q_len, num_qo_heads, num_splits);
}