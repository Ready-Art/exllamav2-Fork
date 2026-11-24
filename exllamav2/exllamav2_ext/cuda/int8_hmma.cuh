// HMMA (tensor-core) INT8 verify attention — flash-attention tiled kernel, HD-split.
// QK^T and PV via nvcuda::wmma m16n16k16 half fragments; INT8 K/V dequantized to
// half on load. Split-KV across CTAs + the int8_decode_merge po/pl contract.
//
// One CTA per (kv-head, split, batch), covering the GQA group's NQ = nq*q_len
// queries (Gemma-4 full layer/rank: nq=8, q_len=4 -> NQ=32). NW = (NQ/16)*HSPLIT
// warps: QT=NQ/16 query tiles (16 rows each) x HSPLIT HD-splits. Each warp owns
// 16 query rows x (HD/HSPLIT) output cols -> O accumulator fits registers.
// QK^T + softmax done by the hdsplit==0 warp of each query tile (S/P to shared);
// all HD-split warps then do PV for their output cols. Online softmax across key
// tiles of TK. m/l/alpha per query row in shared.
//
// wmma m16n16k16 accumulator lane layout: gid=lane/4 owns rows {gid, gid+8};
// x[0,1,4,5]->row gid, x[2,3,6,7]->row gid+8; cols tid*2+{0,1} and tid*2+{8,9}.
#include <mma.h>
using namespace nvcuda;

template <int HD, int TK, int NQ, int HSPLIT>
__global__ void __launch_bounds__((NQ / 16) * HSPLIT * 32, 2)
int8_verify_hmma_kernel(
    const half* __restrict__ q, const int8_t* __restrict__ k_i8,
    const int8_t* __restrict__ v_i8, const half* __restrict__ k_scale,
    const half* __restrict__ v_scale, float* __restrict__ po,
    float* __restrict__ pl, const int* __restrict__ kv_len_dev,
    const int* __restrict__ row_map, int depth, int q_len,
    int num_qo_heads, int num_kv_heads, int num_splits, int batch_size,
    float scale_log2e)
{
    constexpr int QT = NQ / 16;
    constexpr int HDW = HD / HSPLIT;      // HD cols per warp
    constexpr int KSW = HDW / 16;         // O fragments per warp
    constexpr int KS = HD / 16;           // QK k-steps
    constexpr int NS = TK / 16;
    const int wid = blockIdx.x;
    const int total = num_kv_heads * num_splits * batch_size;
    if (wid >= total) return;
    const int kvh = wid % num_kv_heads;
    const int sb = wid / num_kv_heads;
    const int split = sb % num_splits;
    const int b = sb / num_splits;
    const int br = row_map ? row_map[b] : b;
    const int kv_len0 = kv_len_dev[b];
    const int chunk = (kv_len0 + num_splits - 1) / num_splits;
    const int kv_begin = split * chunk;
    int kv_end = kv_begin + chunk;
    if (kv_end > kv_len0) kv_end = kv_len0;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qt = warp / HSPLIT;         // query tile 0..QT-1
    const int hs = warp % HSPLIT;         // HD split 0..HSPLIT-1
    const int group = num_qo_heads / num_kv_heads;
    const int qrow0 = qt * 16;
    const int gid = lane >> 2;
    const int tid = lane & 3;
    const int rowA = gid, rowB = gid + 8;
    const int hd0 = hs * HDW;             // first HD col of this warp

    extern __shared__ half dyn[];
    half* Ks = dyn;
    half* Vs = dyn + TK * HD;
    __shared__ float Ss[QT][16 * TK];
    __shared__ half Ph[QT][16 * TK];
    __shared__ half Qst[QT][16 * 16];
    __shared__ float m_sh[NQ], l_sh[NQ], alpha_sh[NQ];

    if (threadIdx.x < NQ) { m_sh[threadIdx.x] = -1e30f; l_sh[threadIdx.x] = 0.0f; }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[KSW];
    #pragma unroll
    for (int kk = 0; kk < KSW; kk++) wmma::fill_fragment(of[kk], 0.0f);

    const int qiA = qrow0 + rowA, qiB = qrow0 + rowB;
    const int causA = kv_len0 - q_len + (qiA % q_len) + 1;
    const int causB = kv_len0 - q_len + (qiB % q_len) + 1;

    const int8_t* kbase = k_i8 + ((long)br * depth) * num_kv_heads * HD + (long)kvh * HD;
    const int8_t* vbase = v_i8 + ((long)br * depth) * num_kv_heads * HD + (long)kvh * HD;
    const half* skbase = k_scale + (long)br * depth * num_kv_heads + kvh;
    const half* svbase = v_scale + (long)br * depth * num_kv_heads + kvh;

    for (int key = kv_begin; key < kv_end; key += TK)
    {
        const int nu = min(TK, kv_end - key);
        for (int i = threadIdx.x; i < TK * HD; i += (QT * HSPLIT) * 32)
        {
            int t = i / HD, d = i % HD;
            if (t < nu)
            {
                long go = (long)(key + t) * num_kv_heads * HD + d;
                float sk = __half2float(skbase[(long)(key + t) * num_kv_heads]);
                float sv = __half2float(svbase[(long)(key + t) * num_kv_heads]);
                Ks[i] = __float2half((float)kbase[go] * sk);
                Vs[i] = __float2half((float)vbase[go] * sv);
            }
            else { Ks[i] = __float2half(0.0f); Vs[i] = __float2half(0.0f); }
        }
        __syncthreads();

        // QK^T + softmax: only hs==0 warp of each query tile
        if (hs == 0)
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[NS];
            #pragma unroll
            for (int nt = 0; nt < NS; nt++) wmma::fill_fragment(sf[nt], 0.0f);
            #pragma unroll
            for (int kk = 0; kk < KS; kk++)
            {
                // gather this query tile's 16 rows x 16 hd cols (kk) into Qst, scaled
                for (int i = lane; i < 16 * 16; i += 32)
                {
                    int rr = i / 16, dd = i % 16;
                    int qi = qrow0 + rr;
                    int h = qi / q_len, r = qi % q_len;
                    int qh = kvh * group + h;
                    float v = __half2float(q[(((long)b * q_len + r) * num_qo_heads + qh) * HD + kk * 16 + dd]);
                    Qst[qt][i] = __float2half(v * scale_log2e);
                }
                __syncwarp();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf;
                wmma::load_matrix_sync(qf, Qst[qt], 16);
                #pragma unroll
                for (int nt = 0; nt < NS; nt++)
                {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
                    wmma::load_matrix_sync(kf, Ks + nt * 16 * HD + kk * 16, HD);
                    wmma::mma_sync(sf[nt], qf, kf, sf[nt]);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NS; nt++)
                wmma::store_matrix_sync(Ss[qt] + nt * 16, sf[nt], TK, wmma::mem_row_major);
            __syncwarp();

            // per-row online softmax (each lane its 2 rows); write half P to Ph
            {
                float mA = m_sh[qiA], mB = m_sh[qiB];
                float m_newA = mA, m_newB = mB;
                for (int t = 0; t < nu; t++)
                {
                    if (key + t < causA) m_newA = fmaxf(m_newA, Ss[qt][rowA * TK + t]);
                    if (key + t < causB) m_newB = fmaxf(m_newB, Ss[qt][rowB * TK + t]);
                }
                float alphaA = exp2f(mA - m_newA), alphaB = exp2f(mB - m_newB);
                float ltA = 0.0f, ltB = 0.0f;
                for (int t = 0; t < TK; t++)
                {
                    float pA = 0.0f, pB = 0.0f;
                    if (t < nu && key + t < causA)
                    { pA = exp2f(Ss[qt][rowA * TK + t] - m_newA); ltA += pA; }
                    if (t < nu && key + t < causB)
                    { pB = exp2f(Ss[qt][rowB * TK + t] - m_newB); ltB += pB; }
                    Ph[qt][rowA * TK + t] = __float2half(pA);
                    Ph[qt][rowB * TK + t] = __float2half(pB);
                }
                m_sh[qiA] = m_newA; m_sh[qiB] = m_newB;
                l_sh[qiA] = l_sh[qiA] * alphaA + ltA;
                l_sh[qiB] = l_sh[qiB] * alphaB + ltB;
                alpha_sh[qiA] = alphaA; alpha_sh[qiB] = alphaB;
            }
        }
        __syncthreads();

        // rescale O by alpha (per row) — all warps
        float alphaA = alpha_sh[qiA], alphaB = alpha_sh[qiB];
        #pragma unroll
        for (int kk = 0; kk < KSW; kk++)
        {
            of[kk].x[0] *= alphaA; of[kk].x[1] *= alphaA;
            of[kk].x[4] *= alphaA; of[kk].x[5] *= alphaA;
            of[kk].x[2] *= alphaB; of[kk].x[3] *= alphaB;
            of[kk].x[6] *= alphaB; of[kk].x[7] *= alphaB;
        }

        // PV: O (16 x HDW) += P (16 x TK) * V (TK x HDW), this warp's HD cols
        #pragma unroll
        for (int kk = 0; kk < KSW; kk++)
        {
            #pragma unroll
            for (int nt = 0; nt < NS; nt++)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                wmma::load_matrix_sync(pf, Ph[qt] + nt * 16, TK);
                wmma::load_matrix_sync(vf, Vs + nt * 16 * HD + hd0 + kk * 16, HD);
                wmma::mma_sync(of[kk], pf, vf, of[kk]);
            }
        }
        __syncthreads();
    }

    // store UNNORMALIZED partial O + (m,l); merge normalizes across splits.
    const int HR = num_qo_heads * q_len;
    const int hA = qiA / q_len, rA = qiA % q_len;
    const int hB = qiB / q_len, rB = qiB % q_len;
    const int hrA = rA * num_qo_heads + (kvh * group + hA);
    const int hrB = rB * num_qo_heads + (kvh * group + hB);
    long baseA = (((long)b * num_splits + split) * HR + hrA) * HD + hd0;
    long baseB = (((long)b * num_splits + split) * HR + hrB) * HD + hd0;
    #pragma unroll
    for (int kk = 0; kk < KSW; kk++)
    {
        int c0 = tid * 2, c1 = tid * 2 + 8, hd = kk * 16;
        po[baseA + hd + c0]     = of[kk].x[0];
        po[baseA + hd + c0 + 1] = of[kk].x[1];
        po[baseA + hd + c1]     = of[kk].x[4];
        po[baseA + hd + c1 + 1] = of[kk].x[5];
        po[baseB + hd + c0]     = of[kk].x[2];
        po[baseB + hd + c0 + 1] = of[kk].x[3];
        po[baseB + hd + c1]     = of[kk].x[6];
        po[baseB + hd + c1 + 1] = of[kk].x[7];
    }
    if (hs == 0 && tid == 0)
    {
        long miA = (((long)b * num_splits + split) * HR + hrA) * 2;
        long miB = (((long)b * num_splits + split) * HR + hrB) * 2;
        pl[miA + 0] = m_sh[qiA]; pl[miA + 1] = l_sh[qiA];
        pl[miB + 0] = m_sh[qiB]; pl[miB + 1] = l_sh[qiB];
    }
}

// Host launcher (defined in int8_hmma.cu). hd512 verify, NQ=(nq/nkv)*q_len in {16,32}.
void int8_attn_hmma_cuda(
    const half* q, const int8_t* k_i8, const int8_t* v_i8,
    const half* k_scale, const half* v_scale, half* o,
    float* po, float* pl, const int* kv_len_dev, const int* row_map,
    int depth, int batch_size, int q_len, int num_qo_heads, int num_kv_heads,
    int num_splits, float sm_scale, int head_dim, cudaStream_t stream);
