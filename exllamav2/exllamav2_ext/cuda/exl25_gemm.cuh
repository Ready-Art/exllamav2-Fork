#ifndef _exl25_gemm_cuh
#define _exl25_gemm_cuh

// EXL2.5 uniform-bit fast lane.
//
// Serves tensors tagged q_uniform_bits: a single bit-width, uniform group size along K,
// no input permutation. Same packed + shuffled lane layout and scale encoding as EXL2
// (qdq_*.cuh dequant reused verbatim), but the inner loop drops everything that bounds
// the legacy kernel at small m: variable-bit dispatch, group-map loads, data-dependent
// nextgroup chains, scalar perm-gathers, and per-column activation reloads.
//
// Optional per-tensor features (runtime flags, branch only in the preload):
//   had_k = 128 : K-side Hadamard incoherence. Weights were pre-rotated by the orthonormal
//                 block-diagonal H128 at conversion; activations get the UNNORMALIZED
//                 butterfly here (fp32 +/-1 add/sub, 7 shared-mem stages -- fp16 partial
//                 sums overflow on real outlier activations). The 1/128 lives in the
//                 stored weights, so no normalization multiply.
//   awq_scale   : per-input-channel inverse protection scale, applied before the butterfly.
//
// MoE: same r_weights early-exit / mul_r_weights contract as the legacy kernel, so the
// fused q_moe_mlp path serves fast-lane experts unchanged.
//
// Graph-safe: fixed buffers, no host sync, half2 atomicAdd output (legacy contract).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "compat.cuh"
#include "../config.h"
#include "matrix_view.cuh"
#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"

// Vectorized 8-bit dequant (legacy QMODE_8BIT is 0 -> scalar fallback there; the fast lane
// always gets the magic-half form). Natural byte order, no shuffle (shuffle_8bit_4 is id).
__forceinline__ __device__ void exl25_dequant_8bit_8
(
    const uint32_t q_0,
    const uint32_t q_1,
    half2 (&dq)[4]
)
{
    const uint32_t c0 = 0x64006400;
    const half z_ = __float2half_rn(-1024.0f - 128.0f);
    const half2 z = __halves2half2(z_, z_);

    half2_uint32 a0((q_0 & 0x00ff00ff) | c0);        // w0, w2 (+1024)
    half2_uint32 a1(((q_0 >> 8) & 0x00ff00ff) | c0); // w1, w3
    half2_uint32 a2((q_1 & 0x00ff00ff) | c0);        // w4, w6
    half2_uint32 a3(((q_1 >> 8) & 0x00ff00ff) | c0); // w5, w7

    half2 t0 = __hadd2(a0.as_half2, z);
    half2 t1 = __hadd2(a1.as_half2, z);
    half2 t2 = __hadd2(a2.as_half2, z);
    half2 t3 = __hadd2(a3.as_half2, z);

    dq[0] = __halves2half2(__low2half(t0), __low2half(t1));
    dq[1] = __halves2half2(__high2half(t0), __high2half(t1));
    dq[2] = __halves2half2(__low2half(t2), __low2half(t3));
    dq[3] = __halves2half2(__high2half(t2), __high2half(t3));
}

// Register-fed dot products. ALL chunk widths accumulate in fp32: the partial sums
// multiply UNSCALED codes (6-bit spans -32..31) by activations, and the group scale is
// applied only afterwards. Hadamard/AWQ activations spread each outlier channel across
// a whole 128-block at ~1000+ magnitude, so a 16-term fp16 partial reaches ~1e5 > 65504
// -> inf -> NaN on real Gemma-4 activations (randn-scale synthetics never overflowed,
// which is why the original parity gates missed this). fp32 partials are magnitude-robust.

template <int wj>
__forceinline__ __device__ half exl25_dot(half2 (&dq)[wj / 2], const half2 (&a2)[wj / 2], const half g_result, const half qs_h)
{
    float result = {};
    #pragma unroll
    for (int i = 0; i < wj / 2; i++)
    {
        float w0 = __low2float(dq[i]);
        float w1 = __high2float(dq[i]);
        float x0 = __low2float(a2[i]);
        float x1 = __high2float(a2[i]);
        result = fma(w0, x0, result);
        result = fma(w1, x1, result);
    }
    result *= __half2float(qs_h);
    return __hadd(__float2half_rn(result), g_result);
}

typedef void (*fp_exl25_gemm_kernel)
(
    const half*,      // a
    const uint32_t*,  // b_q_weight
    const uint32_t*,  // b_q_scale
    const half*,      // b_q_scale_max
    half*,            // c
    const int,        // size_m
    const int,        // size_n
    const int,        // size_k
    const int,        // groupsize
    const half*,      // awq_scale (NULL = off)
    const int,        // had_k (0 = off, 128 = on)
    const bool,       // clear
    const half*,      // r_weights
    const int         // r_weights_stride
);

template <int bits, int m_count, int block_kn, bool use_r_weights, bool mul_r_weights>
__global__ void exl25_gemm_kernel
(
    const half*      __restrict__ a,
    const uint32_t*  __restrict__ b_q_weight,
    const uint32_t*  __restrict__ b_q_scale,
    const half*      __restrict__ b_q_scale_max,
    half*            __restrict__ c,
    const int size_m,
    const int size_n,
    const int size_k,
    const int groupsize,
    const half*      __restrict__ awq_scale,
    const int had_k,
    const bool clear,
    const half* r_weights,
    const int r_weights_stride
)
{
    // Per-bit chunk geometry (mirrors the legacy lane loops)
    constexpr int chunk_k = (bits == 2) ? 16 : 32;                       // k per outer iteration
    constexpr int jn      = (bits == 8 || bits == 4) ? 4 : (bits == 6 || bits == 2) ? 2 : 1;
    constexpr int wj      = (bits == 8 || bits == 4) ? 8 : (bits == 6 || bits == 2) ? 16 : 32;
    constexpr int rows_j  = wj * bits / 32;                              // packed rows per j
    // Activation register-cache subgroup: wide chunks (3/5-bit, 16 half2/row) use 2 rows
    // to keep registers bounded; narrow chunks cache 4.
    constexpr int sub     = (wj <= 16) ? (m_count < 4 ? m_count : 4) : (m_count < 2 ? m_count : 2);

    MatrixView_half_rw c_(c, size_m, size_n);
    MatrixView_q4_row b_q_scale_(b_q_scale, size_k / groupsize, size_n);

    int t = threadIdx.x;
    int offset_n = blockIdx.x * block_kn * 4;
    int offset_m = blockIdx.y * m_count;
    int offset_k = blockIdx.z * block_kn;
    int m_count_min = min(size_m - offset_m, m_count);
    int end_k = min(offset_k + block_kn, size_k);
    int n = offset_n + t * 4;

    // Routing weights (MoE)

    half_uint16 weights[m_count];
    if constexpr (use_r_weights)
    {
        uint16_t any_w = 0;
        const half* w_ptr = r_weights;
        #pragma unroll
        for (int m = 0; m < m_count; ++m)
        {
            if (m < m_count_min)
            {
                weights[m].as_half = *w_ptr;
                w_ptr += r_weights_stride;
                any_w |= weights[m].as_uint16;
            }
        }
        if (!any_w) return;  // Early exit if all weights are zero -- does not zero output (!!!)
    }

    // Preload activations: vectorized int4 (no permutation in the fast lane), optional
    // AWQ inverse-scale multiply per channel.
    //
    // had/awq tensors preload + butterfly in fp32: real Gemma-4 activations carry
    // ~3000-magnitude outlier channels, and butterfly PARTIAL sums (up to 128 aligned
    // terms; AWQ amplifies small channels up to 4x) overflow fp16's 65504 even when the
    // final sums fit -- fp16 stages gave inf -> inf-inf = NaN -> gibberish decode.
    // Plain uniform tensors keep the original direct-half preload bit-for-bit.

    // Separate fp32 staging + fp16 dot buffers. A float/half union aliases the same
    // bytes; nvcc -O3 miscompiles the cross-member narrowing (the dot loop read
    // uninitialized shared memory -> all-NaN / all-zero outputs). Two arrays, no UB.
    __shared__ float block_af[m_count][block_kn];
    __shared__ half block_a[m_count][block_kn];

    if (had_k || awq_scale)
    {
        if (t < block_kn / 8)
        {
            int kk = offset_k + t * 8;
            if (kk < end_k)
            {
                #pragma unroll
                for (int m = 0; m < m_count; ++m)
                {
                    if (m >= m_count_min) continue;
                    int4 v = *((const int4*) (a + (offset_m + m) * size_k + kk));
                    const half* vh = (const half*) &v;
                    if (awq_scale)
                    {
                        int4 s = *((const int4*) (awq_scale + kk));
                        const half* sh = (const half*) &s;
                        #pragma unroll
                        for (int i = 0; i < 8; i++)
                            block_af[m][t * 8 + i] = __half2float(vh[i]) * __half2float(sh[i]);
                    }
                    else
                    {
                        #pragma unroll
                        for (int i = 0; i < 8; i++)
                            block_af[m][t * 8 + i] = __half2float(vh[i]);
                    }
                }
            }
        }
        __syncthreads();

        // K-side Hadamard butterfly over 128-blocks (unnormalized; weights carry 1/128).
        // All threads participate -- the n-boundary return must come AFTER this, since a
        // live thread's butterfly partner may sit in a column slot past size_n.

        if (had_k)
        {
            const int hb = t >> 7;
            const int he = t & 127;
            #pragma unroll
            for (int h = 1; h < 128; h <<= 1)
            {
                const int pt = (hb << 7) | (he ^ h);
                // 4-row groups keep the butterfly temporaries at 8 registers even for m16.
                #pragma unroll
                for (int mg = 0; mg < m_count; mg += 4)
                {
                    float v[4], p[4];
                    #pragma unroll
                    for (int r = 0; r < 4; r++)
                    {
                        if (mg + r < m_count_min)
                        {
                            v[r] = block_af[mg + r][t];
                            p[r] = block_af[mg + r][pt];
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int r = 0; r < 4; r++)
                    {
                        if (mg + r < m_count_min)
                            block_af[mg + r][t] = (he & h) ? (p[r] - v[r]) : (v[r] + p[r]);
                    }
                    __syncthreads();
                }
            }
        }

        // Narrow to fp16 for the dot loop (separate arrays -- no aliasing).
        #pragma unroll
        for (int m = 0; m < m_count; m++)
            if (m < m_count_min) block_a[m][t] = __float2half_rn(block_af[m][t]);
        __syncthreads();
    }
    else
    {
        if (t < block_kn / 8)
        {
            int kk = offset_k + t * 8;
            if (kk < end_k)
            {
                #pragma unroll
                for (int m = 0; m < m_count; ++m)
                {
                    if (m >= m_count_min) continue;
                    int4 v = *((const int4*) (a + (offset_m + m) * size_k + kk));
                    *((int4*) &block_a[m][t * 8]) = v;
                }
            }
        }
        __syncthreads();
    }

    if (n >= size_n) return;

    // Clear (legacy contract: z=0 clears, every z atomicAdds)

    if (clear && blockIdx.z == 0)
    {
        #pragma unroll
        for (int m = 0; m < m_count; m++)
            if (m < m_count_min) *((uint64_t*) c_.item_ptr(offset_m + m, n)) = 0;
    }

    // Preload scales: arithmetic group indexing, no group map

    half scales[block_kn / 32][4];
    int gcount = (end_k - offset_k) / groupsize;
    int g0 = offset_k / groupsize;
    for (int g = 0; g < gcount; g++)
    {
        int qscales[4];
        b_q_scale_.item4(qscales, g0 + g, n);
        qscales[0]++;
        qscales[1]++;
        qscales[2]++;
        qscales[3]++;
        half maxscale = b_q_scale_max[g0 + g];
        scales[g][0] = __hmul(__int2half_rn(qscales[0] * qscales[0]), maxscale);
        scales[g][1] = __hmul(__int2half_rn(qscales[1] * qscales[1]), maxscale);
        scales[g][2] = __hmul(__int2half_rn(qscales[2] * qscales[2]), maxscale);
        scales[g][3] = __hmul(__int2half_rn(qscales[3] * qscales[3]), maxscale);
    }

    int scales_idx = 0;
    half qs_h[4] = { scales[0][0], scales[0][1], scales[0][2], scales[0][3] };
    int nextgroup = offset_k + groupsize;

    // Weight pointer: single bit-width section, packed rows start at offset_k * bits / 32

    const uint32_t* b_ptr = b_q_weight + (offset_k * bits / 32) * size_n + n;
    half block_c[m_count][4] = {};

    int k = offset_k;
    while (k < end_k)
    {
        if (k == nextgroup)
        {
            scales_idx++;
            qs_h[0] = scales[scales_idx][0];
            qs_h[1] = scales[scales_idx][1];
            qs_h[2] = scales[scales_idx][2];
            qs_h[3] = scales[scales_idx][3];
            nextgroup += groupsize;
        }

        #pragma unroll
        for (int j = 0; j < jn; j++)
        {
            uint32_t q[rows_j][4];
            #pragma unroll
            for (int r = 0; r < rows_j; r++)
            {
                int4 li = *((const int4*) b_ptr);
                b_ptr += size_n;
                q[r][0] = li.x; q[r][1] = li.y; q[r][2] = li.z; q[r][3] = li.w;
            }

            half2 dq[4][wj / 2];
            #pragma unroll
            for (int c = 0; c < 4; c++)
            {
                if constexpr (bits == 8) exl25_dequant_8bit_8(q[0][c], q[1][c], dq[c]);
                if constexpr (bits == 6) dequant_6bit_16(q[0][c], q[1][c], q[2][c], dq[c], size_n);
                if constexpr (bits == 5) dequant_5bit_32(q[0][c], q[1][c], q[2][c], q[3][c], q[4][c], dq[c], size_n);
                if constexpr (bits == 4) dequant_4bit_8(q[0][c], dq[c], size_n);
                if constexpr (bits == 3) dequant_3bit_32(q[0][c], q[1][c], q[2][c], dq[c], size_n);
                if constexpr (bits == 2) dequant_2bit_16(q[0][c], dq[c], size_n);
            }

            // Dequant once, reuse across all 4 columns; activations cached in registers
            // per subgroup (4x fewer shared loads than the legacy per-column dots).
            // MUST unroll: block_c[m0+r] with a runtime m0 loop lands in local memory.

            #pragma unroll
            for (int m0 = 0; m0 < m_count; m0 += sub)
            {
                half2 a2[sub][wj / 2];
                #pragma unroll
                for (int r = 0; r < sub; r++)
                {
                    if (m0 + r < m_count_min)
                    {
                        // (k - offset_k) advances across the block's k chunks; j * wj
                        // selects the chunk within the current 32-k iteration.
                        const half2* ap = (const half2*) &block_a[m0 + r][(k - offset_k) + j * wj];
                        #pragma unroll
                        for (int i = 0; i < wj / 2; i++) a2[r][i] = ap[i];
                    }
                }
                #pragma unroll
                for (int r = 0; r < sub; r++)
                {
                    if (m0 + r >= m_count_min) continue;
                    if constexpr (use_r_weights) { if (!weights[m0 + r].as_uint16) continue; }
                    #pragma unroll
                    for (int c = 0; c < 4; c++)
                        block_c[m0 + r][c] = exl25_dot<wj>(dq[c], a2[r], block_c[m0 + r][c], qs_h[c]);
                }
            }
        }
        k += chunk_k;
    }

    // Accumulate column sums in c (legacy contract)

    #pragma unroll
    for (int m = 0; m < m_count; m++)
    {
        if (m >= m_count_min) continue;
        half2* out = (half2*) c_.item_ptr(offset_m + m, n);
        half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
        half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);

        if constexpr (mul_r_weights)
        {
            half2 w_mul2 = __half2half2(weights[m].as_half);
            result01 = __hmul2(result01, w_mul2);
            result23 = __hmul2(result23, w_mul2);
        }

        atomicAdd(out    , result01);
        atomicAdd(out + 1, result23);
    }
}

// Pickers (instantiations split across exl25_gemm_a.cu / exl25_gemm_b.cu)

fp_exl25_gemm_kernel pick_exl25_kernel_a(int bits, int max_m, int kn, bool r_weights, bool mul_r_weights);
fp_exl25_gemm_kernel pick_exl25_kernel_b(int bits, int max_m, int kn, bool r_weights, bool mul_r_weights);

inline fp_exl25_gemm_kernel pick_exl25_kernel(int bits, int max_m, int kn, bool r_weights, bool mul_r_weights)
{
    if (bits <= 4) return pick_exl25_kernel_a(bits, max_m, kn, r_weights, mul_r_weights);
    return pick_exl25_kernel_b(bits, max_m, kn, r_weights, mul_r_weights);
}

#endif
