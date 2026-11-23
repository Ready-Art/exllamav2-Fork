#include "verify_hd512.cuh"
#include <cuda_pipeline.h>
#include <cstdio>

// hd512 small-q GQA verify attention (FlashDecoding-style split-KV).
//
// Rewritten 2026-11-23 for the MTP verify geometry (q_len <= 4, GQA group 8, kv up to
// ~160k). The previous wmma version ran ~80 GB/s: 16-row tensor-core tiles for q_len=4
// (75% waste), __launch_bounds__(128,1) = 4 warps/SM (cannot hide HBM latency), and
// sequential K->V loads with no overlap. This version:
//   * one CTA per (kv_head, split) serves the WHOLE GQA group x q_len (<= 32 rows),
//     so each KV chunk is read from HBM exactly once (no cross-CTA L2 reliance);
//   * Q lives in registers (64 halves/thread), KV tiles (8 keys) are cp.async
//     double-buffered through 32 KB shared -> 2 CTAs/SM, 16 warps/SM;
//   * CUDA-core fp32 FMAs (the op is bandwidth-bound at small q; tensor cores waste
//     16-row tiles and the register pressure caps occupancy).
// Partials are UNNORMALIZED online-softmax (po, pl=(m,l)) merged by the unchanged
// merge kernel. Graph-safe: device-side kv_len, fixed buffers, no host sync.

#define HDV 512
#define BNV 8
#define ROWS_MAX 32
#define NTHREADS 256
#define DIMS_PER_THREAD (HDV / (NTHREADS / ROWS_MAX))   // 64
#define SMEM_BYTESV (2 * 2 * BNV * HDV * sizeof(half))  // K/V x double buffer = 32 KB

// 1 CTA/SM: the register-resident Q (32 regs) + fp32 output accumulators (64 regs) +
// softmax state need ~130 registers; forcing 2 CTAs/SM (128-reg cap) spills oacc to
// local memory and amplifies HBM traffic ~64x (measured 30 GB/s). 8 warps/SM with
// cp.async double buffering still saturates HBM when splits >= SM count.
__global__ void __launch_bounds__(NTHREADS, 1)
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
    const int gqa = num_qo_heads / num_kv_heads;
    const int rows = q_len * gqa;               // <= ROWS_MAX guarded host-side
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int r = tid / (NTHREADS / ROWS_MAX);  // query row this thread serves (0..31)
    const int d0 = (tid % (NTHREADS / ROWS_MAX)) * DIMS_PER_THREAD;
    const int q_idx = r / gqa;
    const int h = kvh * gqa + (r % gqa);

    extern __shared__ char smem_raw[];
    half* sK = (half*)smem_raw;                 // [2][BNV][HDV]
    half* sV = sK + 2 * BNV * HDV;              // [2][BNV][HDV]

    const long stride_q = (long)num_qo_heads * HDV;
    const long stride_kv = (long)num_kv_heads * HDV;

    const int chunk = (kv_len + num_splits - 1) / num_splits;
    const int kv_begin = split * chunk;
    int kv_end = kv_begin + chunk;
    if (kv_end > kv_len) kv_end = kv_len;
    // Tile count must be CTA-UNIFORM (syncthreads inside the loop); per-row lower-right
    // causality is applied in the score mask below. The last row's bound is kv_len-1.
    const int hi = kv_end;

    // Q into registers, pre-scaled by sm_scale*log2(e) for exp2f softmax.
    half2 qreg[DIMS_PER_THREAD / 2];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD / 2; i++) qreg[i] = __float2half2_rn(0.0f);
    if (r < rows)
    {
        const half* qp = q + (long)q_idx * stride_q + (long)h * HDV + d0;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD / 2; i++)
        {
            half2 hh = *((const half2*)(qp + i * 2));
            float2 f = __half22float2(hh);
            f.x *= scale_log2e; f.y *= scale_log2e;
            qreg[i] = __float22half2_rn(f);
        }
    }

    float oacc[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) oacc[i] = 0.0f;
    float m_i = -1e30f, l_i = 0.0f;

    // cp.async tile load: BNV*HDV halves = 8 KB = 512 x 16B ops / 256 threads = 2 each.
    auto load_tile = [&](half* dst, const half* src, int n0)
    {
        #pragma unroll
        for (int u = 0; u < 2; u++)
        {
            int idx = tid + u * NTHREADS;       // 0..511, 8 halves each
            int key = idx / (HDV / 8);
            int c = (idx % (HDV / 8)) * 8;
            const half* sp = (n0 + key < kv_len)
                ? src + (long)(n0 + key) * stride_kv + (long)kvh * HDV + c
                : nullptr;
            if (sp) __pipeline_memcpy_async(dst + key * HDV + c, sp, 16);
            else *((float4*)(dst + key * HDV + c)) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __pipeline_commit();
    };

    const int ntiles = (hi > kv_begin) ? (hi - kv_begin + BNV - 1) / BNV : 0;
    // Commit-group order is load-bearing: K[0], V[0], K[1]; in-loop V[t+1] then K[t+2].
    // wait_prior(1) then guarantees K[t] and V[t] are ready with K[t+1] still in flight.
    if (ntiles > 0) load_tile(sK, k, kv_begin);
    if (ntiles > 0) load_tile(sV, v, kv_begin);
    if (ntiles > 1) load_tile(sK + BNV * HDV, k, kv_begin + BNV);

    const unsigned rowmask = 0xffffffffu;
    for (int t = 0; t < ntiles; t++)
    {
        int buf = t & 1;
        int n0 = kv_begin + t * BNV;
        // wait for K[t] and V[t]; K[t+1] was issued before V[t] so wait_group 1
        // leaves K[t+1] in flight while K[t] and V[t] are ready.
        if (t + 1 < ntiles) __pipeline_wait_prior(1);
        else __pipeline_wait_prior(0);
        __syncthreads();

        const half* Kt = sK + buf * BNV * HDV;
        const half* Vt = sV + buf * BNV * HDV;

        // scores for this thread's (row, key=t%8 lane assignment): each thread owns
        // ONE key per row iteration; reduce across the 8 threads of the row group.
        float p[BNV];
        float mt = -1e30f;
        #pragma unroll
        for (int j = 0; j < BNV; j++)
        {
            int key = n0 + j;
            float s = 0.0f;
            const half* kp = Kt + j * HDV + d0;
            #pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD / 2; i++)
            {
                float2 kf = __half22float2(*((const half2*)(kp + i * 2)));
                float2 qf = __half22float2(qreg[i]);
                s = fmaf(qf.x, kf.x, s);
                s = fmaf(qf.y, kf.y, s);
            }
            // row-group reduction (8 consecutive lanes)
            s += __shfl_xor_sync(rowmask, s, 1);
            s += __shfl_xor_sync(rowmask, s, 2);
            s += __shfl_xor_sync(rowmask, s, 4);
            bool ok = (r < rows) && (key < hi) && (key <= kv_len - q_len + q_idx);
            if (!ok) s = -1e30f;
            mt = fmaxf(mt, s);
            p[j] = s;
        }
        float mnew = fmaxf(m_i, mt);
        float alpha = exp2f(m_i - mnew);
        float ls = 0.0f;
        #pragma unroll
        for (int j = 0; j < BNV; j++) { p[j] = exp2f(p[j] - mnew); ls += p[j]; }
        l_i = l_i * alpha + ls;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) oacc[i] *= alpha;
        #pragma unroll
        for (int j = 0; j < BNV; j++)
        {
            float pj = p[j];
            const half* vp = Vt + j * HDV + d0;
            #pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; i++)
                oacc[i] = fmaf(pj, __half2float(vp[i]), oacc[i]);
        }
        m_i = mnew;
        __syncthreads();

        // issue V[t+1] then K[t+2] (order matters for the wait_prior(1) invariant)
        if (t + 1 < ntiles) load_tile(sV + ((t + 1) & 1) * BNV * HDV, v, n0 + BNV);
        if (t + 2 < ntiles) load_tile(sK + (t & 1) * BNV * HDV, k, n0 + 2 * BNV);
    }

    // epilogue: unnormalized partials for the merge kernel
    if (r < rows)
    {
        long base = ((long)(split * q_len + q_idx) * num_qo_heads + h) * HDV;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) po[base + d0 + i] = oacc[i];
        if (d0 == 0)
        {
            long idx = ((long)(split * q_len + q_idx) * num_qo_heads + h) * 2;
            pl[idx + 0] = m_i;
            pl[idx + 1] = l_i;
        }
    }
}

// Merge the split partials with a global log-sum-exp. One BLOCK per (row, head); the
// 512 output dims are striped across the block's threads so every po/pl read is fully
// coalesced. blockDim.x must be a multiple of 32 and divide 512.
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

    __shared__ float s_w[256];              // split weights (num_splits <= 256)
    __shared__ float s_gm;
    __shared__ float s_den;

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

    if (d < num_splits)
    {
        long mi = ((long)(d * q_len + r) * num_qo_heads + h) * 2;
        s_w[d] = exp2f(pl[mi + 0] - gm);
        if (d == 0) s_den = 0.0f;
    }
    __syncthreads();
    if (d < num_splits)
    {
        long mi = ((long)(d * q_len + r) * num_qo_heads + h) * 2;
        atomicAdd(&s_den, s_w[d] * pl[mi + 1]);
    }
    __syncthreads();
    const float inv = (s_den > 0.0f) ? 1.0f / s_den : 0.0f;

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
    // rows = q_len * GQA must fit the register-resident query tile
    if (q_len * (num_qo_heads / num_kv_heads) > ROWS_MAX) return;
    cudaFuncSetAttribute(verify_hd512_partial_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTESV);
    float scale_log2e = sm_scale * 1.4426950408889634f;
    dim3 grid(num_kv_heads, num_splits, 1);
    verify_hd512_partial_kernel<<<grid, NTHREADS, SMEM_BYTESV, stream>>>
        (q, k, v, po, pl, q_len, kv_len_dev, num_qo_heads, num_kv_heads, num_splits, scale_log2e);

    long blocks = (long)q_len * num_qo_heads;
    verify_hd512_merge_kernel<<<(int)blocks, HDV, 0, stream>>>
        (po, pl, o, q_len, num_qo_heads, num_splits);
}
