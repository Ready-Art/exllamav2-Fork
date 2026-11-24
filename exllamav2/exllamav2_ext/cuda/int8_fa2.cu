#include "int8_fa2.cuh"
#include "int8_decode.cuh"

// FA2-style pipelined INT8 verify attention, hd512, NQ = (nq/nkv)*q_len in {16,32}.
//
// Why this exists: the warp kernel is CUDA-core compute-bound (~35 GB/s effective)
// and the HMMA kernel runs QK^T on only the hs==0 warp of each query tile (2 of 8
// warps), so its QK phase is ~4x longer than the PV phase that follows it. Both
// lose to FlashInfer's fp16 paged prefill at long context, which is why INT8 KV
// made verify SLOWER there despite halving the bytes.
//
// Here every warp does useful work in BOTH phases:
//   QK: warp w owns keys [w*8, w*8+8) of the TK=32 tile  -> mma m16n8k16, N=8 full
//   PV: warp w owns HD cols [w*128, (w+1)*128)           -> 16 n-tiles, K=2 steps
// Both are MT*32 mma per warp per tile (MT = NQ/16), so the phases are balanced.
// Cross-warp online softmax reduces only per-row m/l partials through smem (never
// the full S tile), which is what keeps shared memory at ~99 KB for hd512.
//
// K/V stream as INT8 through double-buffered cp.async tiles and are dequantized to
// half in registers at fragment-load time. No half smem tiles: at hd512 a half
// K+V tile pair costs 4x the int8 one and would not fit alongside Q.
//
// Graph-safe and numerically identical in scheme to int8_attn_fast: per-(token,head)
// fp16 scales, exp2 online softmax, unnormalized fp32 partials + (m,l) merged by
// int8_merge_cuda. Per-row causal frontier kv_len_dev[b] - q_len + r + 1.

#define FA2_NW 4
#define FA2_TK 32
#define FA2_HD 512
#define FA2_KPW (FA2_TK / FA2_NW)      // 8 keys per warp in QK
#define FA2_HPW (FA2_HD / FA2_NW)      // 128 HD cols per warp in PV

__device__ __forceinline__ void fa2_cp16(void* smem, const void* gmem, bool valid)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int sz = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(sz));
}

__device__ __forceinline__ void fa2_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void fa2_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void fa2_mma(float& d0, float& d1, float& d2, float& d3,
                                        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned fa2_h2i(half a, half b)
{
    __half2 h = __halves2half2(a, b);
    return *(unsigned*)&h;
}

// int8 pair -> dequantized half2 (bit-identical to float multiply then round:
// both operands are exact halves, so the product rounds once either way).
__device__ __forceinline__ unsigned fa2_deq2(int a, int b, half2 sc)
{
    __half2 h = __halves2half2(__int2half_rn(a), __int2half_rn(b));
    h = __hmul2(h, sc);
    return *(unsigned*)&h;
}

template <int NQ>
__global__ void __launch_bounds__(FA2_NW * 32, 1)
int8_attn_fa2_kernel(
    const half* __restrict__ q,
    const int8_t* __restrict__ k_i8,
    const int8_t* __restrict__ v_i8,
    const half* __restrict__ k_scale,
    const half* __restrict__ v_scale,
    float* __restrict__ po,
    float* __restrict__ pl,
    const int* __restrict__ kv_len_dev,
    const int* __restrict__ row_map,
    int depth, int q_len, int num_qo_heads, int num_kv_heads,
    int num_splits, int batch_size, float scale_log2e)
{
    constexpr int MT = NQ / 16;            // mma M-tiles
    constexpr int NT = FA2_HPW / 8;        // PV n-tiles per warp (16)
    constexpr int KS = FA2_HD / 16;        // QK k-steps (32)
    constexpr int ROWS_PER_WARP = NQ / FA2_NW;

    // sm86 caps a block at 101376 B (1 KB of the 102400 B/SM is driver-reserved), so
    // K is double-buffered but V is single: V is not read until PV, so its cp.async
    // issued at the top of the tile overlaps QK + softmax and is waited on before PV.
    extern __shared__ char smem_raw[];
    int8_t* Ks = (int8_t*)smem_raw;                                  // 2 * TK * HD
    int8_t* Vs = Ks + 2 * FA2_TK * FA2_HD;                           // TK * HD
    half*   Qs = (half*)(Vs + FA2_TK * FA2_HD);                      // NQ * HD
    half*   Ps = Qs + NQ * FA2_HD;                                   // NQ * TK
    float*  m_run = (float*)(Ps + NQ * FA2_TK);
    float*  l_run = m_run + NQ;
    float*  m_part = l_run + NQ;                                     // NW * NQ
    float*  l_part = m_part + FA2_NW * NQ;
    float*  m_new = l_part + FA2_NW * NQ;
    float*  alpha = m_new + NQ;
    half*   ks_s = (half*)(alpha + NQ);                              // 2 * TK (double buffered)
    half*   vs_s = ks_s + 2 * FA2_TK;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;             // mma group id (row / key-n)
    const int tig = lane & 3;              // thread in group (col / k)

    const int wid = blockIdx.x;
    const int total = num_kv_heads * num_splits * batch_size;
    if (wid >= total) return;
    const int kvh = wid % num_kv_heads;
    const int sb = wid / num_kv_heads;
    const int split = sb % num_splits;
    const int b = sb / num_splits;
    const int br = row_map ? row_map[b] : b;
    const int group = num_qo_heads / num_kv_heads;
    const int kv_len0 = kv_len_dev[b];

    const int chunk = (kv_len0 + num_splits - 1) / num_splits;
    const int kv_begin = split * chunk;
    int kv_end = kv_begin + chunk;
    if (kv_end > kv_len0) kv_end = kv_len0;

    const long kstride = (long)num_kv_heads * FA2_HD;
    const int8_t* kbase = k_i8 + ((long)br * depth) * kstride + (long)kvh * FA2_HD;
    const int8_t* vbase = v_i8 + ((long)br * depth) * kstride + (long)kvh * FA2_HD;
    const half* skbase = k_scale + (long)br * depth * num_kv_heads + kvh;
    const half* svbase = v_scale + (long)br * depth * num_kv_heads + kvh;

    // ---- Q into smem (row qi = h*q_len + r within this kv head's GQA group) ----
    {
        const int chunks = NQ * FA2_HD * 2 / 16;
        for (int c = tid; c < chunks; c += FA2_NW * 32)
        {
            int qi = c / (FA2_HD * 2 / 16);
            int off = (c % (FA2_HD * 2 / 16)) * 16;
            int h = qi / q_len, r = qi % q_len;
            const half* src = q + (((long)b * q_len + r) * num_qo_heads + kvh * group + h) * FA2_HD + off / 2;
            fa2_cp16(Qs + qi * FA2_HD + off / 2, src, true);
        }
        fa2_commit();
    }

    // ---- per-row causal frontiers for this lane's accumulator rows ----
    float causA[MT], causB[MT];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
    {
        int qiA = mt * 16 + gid, qiB = mt * 16 + gid + 8;
        causA[mt] = (float)(kv_len0 - q_len + (qiA % q_len) + 1);
        causB[mt] = (float)(kv_len0 - q_len + (qiB % q_len) + 1);
    }

    // ---- O accumulators (this warp's HD slice) ----
    float oacc[MT][NT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
        #pragma unroll
        for (int nt = 0; nt < NT; nt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) oacc[mt][nt][i] = 0.0f;

    if (tid < NQ) { m_run[tid] = -1e30f; l_run[tid] = 0.0f; }

    // ---- prologue: first K/V tile ----
    const int ntiles = (kv_end > kv_begin) ? (kv_end - kv_begin + FA2_TK - 1) / FA2_TK : 0;
    const int chunks = FA2_TK * FA2_HD / 16;
    auto issue_k = [&](int t, int buf)
    {
        int key0 = kv_begin + t * FA2_TK;
        int nu = min(FA2_TK, kv_end - key0);
        for (int c = tid; c < chunks; c += FA2_NW * 32)
        {
            int kk = c / (FA2_HD / 16);
            int off = (c % (FA2_HD / 16)) * 16;
            bool ok = kk < nu;
            long g = (long)(key0 + (ok ? kk : 0)) * kstride + off;
            fa2_cp16(Ks + buf * FA2_TK * FA2_HD + kk * FA2_HD + off, kbase + g, ok);
        }
        if (tid < FA2_TK)
        {
            bool ok = tid < nu;
            ks_s[buf * FA2_TK + tid] = ok ? skbase[(long)(key0 + tid) * num_kv_heads] : __float2half(0.0f);
        }
    };
    auto issue_v = [&](int t)
    {
        int key0 = kv_begin + t * FA2_TK;
        int nu = min(FA2_TK, kv_end - key0);
        for (int c = tid; c < chunks; c += FA2_NW * 32)
        {
            int kk = c / (FA2_HD / 16);
            int off = (c % (FA2_HD / 16)) * 16;
            bool ok = kk < nu;
            long g = (long)(key0 + (ok ? kk : 0)) * kstride + off;
            fa2_cp16(Vs + kk * FA2_HD + off, vbase + g, ok);
        }
        if (tid < FA2_TK)
        {
            bool ok = tid < nu;
            vs_s[tid] = ok ? svbase[(long)(key0 + tid) * num_kv_heads] : __float2half(0.0f);
        }
    };

    if (ntiles > 0) { issue_k(0, 0); fa2_commit(); }
    fa2_wait<0>();
    __syncthreads();

    for (int t = 0; t < ntiles; t++)
    {
        const int buf = t & 1;
        const int key0 = kv_begin + t * FA2_TK;
        const int nu = min(FA2_TK, kv_end - key0);
        const int8_t* Kt = Ks + buf * FA2_TK * FA2_HD;
        const int8_t* Vt = Vs;

        // K(t+1) into the other buffer and V(t) into the single V buffer: both overlap
        // the QK + softmax below and are waited on just before PV. Overwriting V(t-1)
        // is safe -- the end-of-tile barrier guarantees PV(t-1) has finished.
        if (t + 1 < ntiles) issue_k(t + 1, buf ^ 1);
        issue_v(t);
        fa2_commit();

        // ---------------- QK^T: this warp's 8 keys ----------------
        float s[MT][4];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) s[mt][i] = 0.0f;

        const int kw = warp * FA2_KPW + gid;      // key index within the tile
        const half2 ksc = __half2half2(ks_s[buf * FA2_TK + kw]);
        #pragma unroll
        for (int ks = 0; ks < KS; ks++)
        {
            const int hd0 = ks * 16 + tig * 2;
            // B = K^T: lane holds K[key=kw][hd0..hd0+1] and [hd0+8..hd0+9]
            const int8_t* kp = Kt + kw * FA2_HD + hd0;
            unsigned b0 = fa2_deq2(kp[0], kp[1], ksc);
            unsigned b1 = fa2_deq2(kp[8], kp[9], ksc);
            #pragma unroll
            for (int mt = 0; mt < MT; mt++)
            {
                const half* qa = Qs + (mt * 16 + gid) * FA2_HD + hd0;
                const half* qb = Qs + (mt * 16 + gid + 8) * FA2_HD + hd0;
                unsigned a0 = *(const unsigned*)(qa);
                unsigned a1 = *(const unsigned*)(qb);
                unsigned a2 = *(const unsigned*)(qa + 8);
                unsigned a3 = *(const unsigned*)(qb + 8);
                fa2_mma(s[mt][0], s[mt][1], s[mt][2], s[mt][3], a0, a1, a2, a3, b0, b1);
            }
        }

        // Apply sm_scale * log2(e) to the fp32 accumulator rather than pre-scaling Q in
        // half: same result as the reference (which scales in fp32) without rounding the
        // scale itself to fp16, and it costs MT*4 multiplies instead of NQ*HD/threads.
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) s[mt][i] *= scale_log2e;

        // ---------------- online softmax (cross-warp m/l reduction) ----------------
        const int kcl = warp * FA2_KPW + tig * 2;            // this lane's 2 keys, tile-local
        const int gkey0 = key0 + kcl;                        // ... and global
        float lmaxA[MT], lmaxB[MT];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
        {
            float a0 = (gkey0 + 0 < causA[mt] && kcl + 0 < nu) ? s[mt][0] : -1e30f;
            float a1 = (gkey0 + 1 < causA[mt] && kcl + 1 < nu) ? s[mt][1] : -1e30f;
            float b0 = (gkey0 + 0 < causB[mt] && kcl + 0 < nu) ? s[mt][2] : -1e30f;
            float b1 = (gkey0 + 1 < causB[mt] && kcl + 1 < nu) ? s[mt][3] : -1e30f;
            lmaxA[mt] = fmaxf(a0, a1);
            lmaxB[mt] = fmaxf(b0, b1);
            // reduce over the 4 lanes of the group -> warp's 8 keys
            lmaxA[mt] = fmaxf(lmaxA[mt], __shfl_xor_sync(0xffffffffu, lmaxA[mt], 1));
            lmaxA[mt] = fmaxf(lmaxA[mt], __shfl_xor_sync(0xffffffffu, lmaxA[mt], 2));
            lmaxB[mt] = fmaxf(lmaxB[mt], __shfl_xor_sync(0xffffffffu, lmaxB[mt], 1));
            lmaxB[mt] = fmaxf(lmaxB[mt], __shfl_xor_sync(0xffffffffu, lmaxB[mt], 2));
            if (tig == 0)
            {
                m_part[warp * NQ + mt * 16 + gid] = lmaxA[mt];
                m_part[warp * NQ + mt * 16 + gid + 8] = lmaxB[mt];
            }
        }
        __syncthreads();

        // row-owner warp combines the tile max with the running max
        #pragma unroll
        for (int i = 0; i < ROWS_PER_WARP; i++)
        {
            int row = warp * ROWS_PER_WARP + i;
            float tm = m_part[row];
            #pragma unroll
            for (int w = 1; w < FA2_NW; w++) tm = fmaxf(tm, m_part[w * NQ + row]);
            float mo = m_run[row];
            float mn = fmaxf(mo, tm);
            m_new[row] = mn;
            alpha[row] = exp2f(mo - mn);
        }
        __syncthreads();

        // rescale O, emit P, accumulate local l
        float lsumA[MT], lsumB[MT];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
        {
            float alA = alpha[mt * 16 + gid];
            float alB = alpha[mt * 16 + gid + 8];
            float mnA = m_new[mt * 16 + gid];
            float mnB = m_new[mt * 16 + gid + 8];
            #pragma unroll
            for (int nt = 0; nt < NT; nt++)
            {
                oacc[mt][nt][0] *= alA; oacc[mt][nt][1] *= alA;
                oacc[mt][nt][2] *= alB; oacc[mt][nt][3] *= alB;
            }
            float p0 = (gkey0 + 0 < causA[mt] && kcl + 0 < nu) ? exp2f(s[mt][0] - mnA) : 0.0f;
            float p1 = (gkey0 + 1 < causA[mt] && kcl + 1 < nu) ? exp2f(s[mt][1] - mnA) : 0.0f;
            float p2 = (gkey0 + 0 < causB[mt] && kcl + 0 < nu) ? exp2f(s[mt][2] - mnB) : 0.0f;
            float p3 = (gkey0 + 1 < causB[mt] && kcl + 1 < nu) ? exp2f(s[mt][3] - mnB) : 0.0f;
            Ps[(mt * 16 + gid) * FA2_TK + kcl] = __float2half(p0);
            Ps[(mt * 16 + gid) * FA2_TK + kcl + 1] = __float2half(p1);
            Ps[(mt * 16 + gid + 8) * FA2_TK + kcl] = __float2half(p2);
            Ps[(mt * 16 + gid + 8) * FA2_TK + kcl + 1] = __float2half(p3);
            lsumA[mt] = p0 + p1;
            lsumB[mt] = p2 + p3;
            lsumA[mt] += __shfl_xor_sync(0xffffffffu, lsumA[mt], 1);
            lsumA[mt] += __shfl_xor_sync(0xffffffffu, lsumA[mt], 2);
            lsumB[mt] += __shfl_xor_sync(0xffffffffu, lsumB[mt], 1);
            lsumB[mt] += __shfl_xor_sync(0xffffffffu, lsumB[mt], 2);
            if (tig == 0)
            {
                l_part[warp * NQ + mt * 16 + gid] = lsumA[mt];
                l_part[warp * NQ + mt * 16 + gid + 8] = lsumB[mt];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ROWS_PER_WARP; i++)
        {
            int row = warp * ROWS_PER_WARP + i;
            float ls = l_part[row];
            #pragma unroll
            for (int w = 1; w < FA2_NW; w++) ls += l_part[w * NQ + row];
            l_run[row] = l_run[row] * alpha[row] + ls;
            m_run[row] = m_new[row];
        }
        __syncthreads();

        // ---------------- PV: this warp's 128 HD cols ----------------
        // V(t) (and K(t+1)) must have landed; this is the only wait in the tile.
        fa2_wait<0>();
        __syncthreads();
        const int hdw = warp * FA2_HPW;
        #pragma unroll
        for (int ks = 0; ks < FA2_TK / 16; ks++)
        {
            const int k0 = ks * 16 + tig * 2;
            const half sv0 = vs_s[k0];
            const half sv1 = vs_s[k0 + 1];
            const half sv8 = vs_s[k0 + 8];
            const half sv9 = vs_s[k0 + 9];
            #pragma unroll
            for (int nt = 0; nt < NT; nt++)
            {
                const int hd = hdw + nt * 8 + gid;
                // B = V (K=keys x N=hd): lane holds V[k0][hd], V[k0+1][hd], V[k0+8][hd], V[k0+9][hd]
                unsigned b0 = fa2_h2i(__hmul(__int2half_rn(Vt[k0 * FA2_HD + hd]), sv0),
                                      __hmul(__int2half_rn(Vt[(k0 + 1) * FA2_HD + hd]), sv1));
                unsigned b1 = fa2_h2i(__hmul(__int2half_rn(Vt[(k0 + 8) * FA2_HD + hd]), sv8),
                                      __hmul(__int2half_rn(Vt[(k0 + 9) * FA2_HD + hd]), sv9));
                #pragma unroll
                for (int mt = 0; mt < MT; mt++)
                {
                    const half* pa = Ps + (mt * 16 + gid) * FA2_TK + k0;
                    const half* pb = Ps + (mt * 16 + gid + 8) * FA2_TK + k0;
                    unsigned a0 = *(const unsigned*)(pa);
                    unsigned a1 = *(const unsigned*)(pb);
                    unsigned a2 = *(const unsigned*)(pa + 8);
                    unsigned a3 = *(const unsigned*)(pb + 8);
                    fa2_mma(oacc[mt][nt][0], oacc[mt][nt][1], oacc[mt][nt][2], oacc[mt][nt][3],
                            a0, a1, a2, a3, b0, b1);
                }
            }
        }

        // PV(t) must finish before the next tile overwrites the single V buffer, and
        // before P is rewritten by the next tile's softmax.
        __syncthreads();
    }

    // ---------------- unnormalized partials + (m,l) ----------------
    const int HR = num_qo_heads * q_len;
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
    {
        int qiA = mt * 16 + gid, qiB = mt * 16 + gid + 8;
        int hA = qiA / q_len, rA = qiA % q_len;
        int hB = qiB / q_len, rB = qiB % q_len;
        int hrA = rA * num_qo_heads + kvh * group + hA;
        int hrB = rB * num_qo_heads + kvh * group + hB;
        long baseA = (((long)b * num_splits + split) * HR + hrA) * FA2_HD + warp * FA2_HPW;
        long baseB = (((long)b * num_splits + split) * HR + hrB) * FA2_HD + warp * FA2_HPW;
        #pragma unroll
        for (int nt = 0; nt < NT; nt++)
        {
            int c = nt * 8 + tig * 2;
            po[baseA + c] = oacc[mt][nt][0];
            po[baseA + c + 1] = oacc[mt][nt][1];
            po[baseB + c] = oacc[mt][nt][2];
            po[baseB + c + 1] = oacc[mt][nt][3];
        }
        if (tig == 0)
        {
            long miA = (((long)b * num_splits + split) * HR + hrA) * 2;
            long miB = (((long)b * num_splits + split) * HR + hrB) * 2;
            pl[miA + 0] = m_run[qiA]; pl[miA + 1] = l_run[qiA];
            pl[miB + 0] = m_run[qiB]; pl[miB + 1] = l_run[qiB];
        }
    }
}

template <int NQ>
static void launch_fa2(
    const half* q, const int8_t* k8, const int8_t* v8, const half* ks, const half* vs,
    float* po, float* pl, const int* kvl, const int* row_map,
    int depth, int qlen, int nq, int nkv, int splits, int B, float sc, cudaStream_t stream)
{
    constexpr int smem = 2 * FA2_TK * FA2_HD              // K int8, double buffered
                       + FA2_TK * FA2_HD                  // V int8, single (prefetched)
                       + NQ * FA2_HD * 2                  // Q half
                       + NQ * FA2_TK * 2                  // P half
                       + NQ * 4 * 4                       // m_run, l_run, m_new, alpha
                       + FA2_NW * NQ * 4 * 2              // m_part, l_part
                       + 2 * FA2_TK * 2                   // k scales, double buffered
                       + FA2_TK * 2;                      // v scales, single
    static_assert(smem <= 101376, "sm86 per-block shared memory optin limit");
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(int8_attn_fa2_kernel<NQ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    int total = nkv * splits * B;
    int8_attn_fa2_kernel<NQ><<<total, FA2_NW * 32, smem, stream>>>(
        q, k8, v8, ks, vs, po, pl, kvl, row_map, depth, qlen, nq, nkv, splits, B, sc);
}

void int8_attn_fa2_cuda(
    const half* q, const int8_t* k_i8, const int8_t* v_i8,
    const half* k_scale, const half* v_scale, half* o,
    float* po, float* pl, const int* kv_len_dev, const int* row_map,
    int depth, int batch_size, int q_len, int num_qo_heads, int num_kv_heads,
    int num_splits, float sm_scale, int head_dim, cudaStream_t stream)
{
    if (head_dim != 512 || num_kv_heads <= 0 || num_qo_heads % num_kv_heads != 0) return;
    int group = num_qo_heads / num_kv_heads;
    int nq_rt = group * q_len;
    float scale_log2e = sm_scale * 1.4426950408889634f;
    if (nq_rt == 16)
        launch_fa2<16>(q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev, row_map,
                       depth, q_len, num_qo_heads, num_kv_heads, num_splits, batch_size,
                       scale_log2e, stream);
    else if (nq_rt == 32)
        launch_fa2<32>(q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev, row_map,
                       depth, q_len, num_qo_heads, num_kv_heads, num_splits, batch_size,
                       scale_log2e, stream);
    else
        return;   // unsupported geometry -- caller falls back
    int HR = num_qo_heads * q_len;
    int8_merge_cuda(po, pl, o, HR, batch_size, num_splits, 512, stream);
}
