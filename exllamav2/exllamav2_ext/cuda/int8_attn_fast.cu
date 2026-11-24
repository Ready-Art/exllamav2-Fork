#include "int8_attn_fast.cuh"
#include "int8_decode.cuh"

// Fast INT8 KV attention (decode q_len=1 AND verify q_len>1 in one kernel).
//
// Root cause of the old warp-per-CTA family's slowness (P10b gate, 2026-11-23):
// sm86 caps resident CTAs per SM at 16, so 32-thread CTAs hard-cap occupancy at
// 16 warps/SM (33%) regardless of registers -- too few in-flight loads to hide
// memory latency (~52-81 GB/s vs FI 340-530). Two fixes here:
//   1. WARPS_PER_CTA=4 independent warps per 128-thread CTA (no shared memory,
//      no syncs -- each warp owns one (head-row, split, batch) work item), so
//      the 16-CTA cap now allows up to 64 warps/SM (register-limited below).
//   2. U=4 multi-key unroll: four independent dot-product chains per warp with
//      interleaved shfl reductions -> 4x the memory-level parallelism per warp.
// Adjacent warps in a CTA share the same split chunk and (via GQA) the same kv
// head bytes, so L2 absorbs the cross-warp reuse exactly like the old kernel.
//
// Graph-safe: device-side kv_len frontier, caller-owned persistent partials,
// optional logical->physical row_map. Same numerical scheme as int8_decode
// (per-(token,head) fp16 scales, exp2 online softmax, fp32 partials + merge).

#define WARPS_PER_CTA 4
// U=1 + 8 CTAs/SM (32 warps, 67% occupancy, ~48 regs, no spill) measured best of
// the warp family on sm86 (standalone sweep 2026-11-24): decode ~130 GB/s, verify
// ~35 GB/s effective. Multi-row verify is COMPUTE-bound (per-key dot + 5-shfl
// reduction + exp2 online softmax on CUDA cores), not memory-bound -- key-outer
// rows-inner (no redundant reads) and shared-tile variants are no faster. Matching
// FlashInfer's ~500 GB/s verify needs an HMMA/tensor-core tiled kernel (future work).
#define U 1

template <int HD>
__global__ void __launch_bounds__(128, 8)
int8_attn_fast_kernel
(
    const half* __restrict__ q,
    const int8_t* __restrict__ k_i8,
    const int8_t* __restrict__ v_i8,
    const half* __restrict__ k_scale,
    const half* __restrict__ v_scale,
    float* __restrict__ po,
    float* __restrict__ pl,
    const int* __restrict__ kv_len_dev,
    const int* __restrict__ row_map,
    int depth,
    int q_len,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    int batch_size,
    float scale_log2e
)
{
    constexpr int DPL = HD / 32;
    const int HR = num_qo_heads * q_len;
    const int wid = blockIdx.x * WARPS_PER_CTA + (threadIdx.x >> 5);
    const int total = HR * num_splits * batch_size;
    if (wid >= total) return;
    // hr-major within a (split, batch) group: adjacent warps share kv-head bytes.
    const int hr = wid % HR;
    const int sb = wid / HR;
    const int split = sb % num_splits;
    const int b = sb / num_splits;
    const int h = hr % num_qo_heads;
    const int r = hr / num_qo_heads;
    const int br = row_map ? row_map[b] : b;
    // Verify: row r's causal frontier is kv_len - q_len + r + 1 (kv_len_dev holds
    // the frontier AFTER all q_len rows were written). Decode (q_len=1): kv_len.
    const int kv_len = kv_len_dev[b] - q_len + r + 1;
    const int lane = threadIdx.x & 31;
    const int d0 = lane * DPL;
    const int kvh = h / (num_qo_heads / num_kv_heads);

    const int chunk = (kv_len + num_splits - 1) / num_splits;
    const int kv_begin = split * chunk;
    int kv_end = kv_begin + chunk;
    if (kv_end > kv_len) kv_end = kv_len;

    float qf[DPL];
    {
        const half* qp = q + (((long)b * q_len + r) * num_qo_heads + h) * HD + d0;
        #pragma unroll
        for (int i = 0; i < DPL; i++) qf[i] = __half2float(qp[i]) * scale_log2e;
    }

    float oacc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; i++) oacc[i] = 0.0f;
    float m_i = -1e30f, l_i = 0.0f;

    const int8_t* kbase = k_i8 + ((long)br * depth) * num_kv_heads * HD + (long)kvh * HD + d0;
    const int8_t* vbase = v_i8 + ((long)br * depth) * num_kv_heads * HD + (long)kvh * HD + d0;
    const half* skbase = k_scale + (long)br * depth * num_kv_heads + kvh;
    const half* svbase = v_scale + (long)br * depth * num_kv_heads + kvh;

    for (int key = kv_begin; key < kv_end; key += U)
    {
        int8_t kr[U][DPL], vr[U][DPL];
        float sk[U], sv[U], s[U];
        const int nu = min(U, kv_end - key);
        #pragma unroll
        for (int u = 0; u < U; u++)
        {
            if (u < nu)
            {
                const int8_t* kp = kbase + (long)(key + u) * num_kv_heads * HD;
                const int8_t* vp = vbase + (long)(key + u) * num_kv_heads * HD;
                #pragma unroll
                for (int i = 0; i < DPL; i += 4)
                {
                    *(int*)(kr[u] + i) = *(const int*)(kp + i);
                    *(int*)(vr[u] + i) = *(const int*)(vp + i);
                }
                sk[u] = __half2float(skbase[(long)(key + u) * num_kv_heads]);
                sv[u] = __half2float(svbase[(long)(key + u) * num_kv_heads]);
            }
        }
        // U independent dot products; the shfl chains interleave (MLP).
        #pragma unroll
        for (int u = 0; u < U; u++)
        {
            s[u] = 0.0f;
            if (u < nu)
            {
                #pragma unroll
                for (int i = 0; i < DPL; i++)
                    s[u] = fmaf(qf[i], (float)kr[u][i] * sk[u], s[u]);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            #pragma unroll
            for (int u = 0; u < U; u++)
                s[u] += __shfl_xor_sync(0xffffffffu, s[u], off);
        }
        // Online softmax over the U scores (sequential, cheap vs the loads).
        #pragma unroll
        for (int u = 0; u < U; u++)
        {
            if (u < nu)
            {
                float mnew = fmaxf(m_i, s[u]);
                float alpha = exp2f(m_i - mnew);
                float p = exp2f(s[u] - mnew);
                l_i = l_i * alpha + p;
                #pragma unroll
                for (int i = 0; i < DPL; i++)
                    oacc[i] = oacc[i] * alpha + p * ((float)vr[u][i] * sv[u]);
                m_i = mnew;
            }
        }
    }

    long pbase = (((long)b * num_splits + split) * HR + hr) * HD;
    #pragma unroll
    for (int i = 0; i < DPL; i++) po[pbase + d0 + i] = oacc[i];
    if (lane == 0)
    {
        long mi = (((long)b * num_splits + split) * HR + hr) * 2;
        pl[mi + 0] = m_i;
        pl[mi + 1] = l_i;
    }
}

void int8_attn_fast_cuda
(
    const half* q,
    const int8_t* k_i8,
    const int8_t* v_i8,
    const half* k_scale,
    const half* v_scale,
    half* o,
    float* po,
    float* pl,
    const int* kv_len_dev,
    const int* row_map,
    int depth,
    int batch_size,
    int q_len,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float sm_scale,
    int head_dim,
    cudaStream_t stream
)
{
    if (num_splits > 2048) return;
    float scale_log2e = sm_scale * 1.4426950408889634f;
    int HR = num_qo_heads * q_len;
    int total = HR * num_splits * batch_size;
    dim3 grid((total + WARPS_PER_CTA - 1) / WARPS_PER_CTA, 1, 1);
    if (head_dim == 512)
    {
        int8_attn_fast_kernel<512><<<grid, 128, 0, stream>>>
            (q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev, row_map,
             depth, q_len, num_qo_heads, num_kv_heads, num_splits, batch_size,
             scale_log2e);
        int8_merge_cuda(po, pl, o, HR, batch_size, num_splits, 512, stream);
    }
    else if (head_dim == 256)
    {
        int8_attn_fast_kernel<256><<<grid, 128, 0, stream>>>
            (q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev, row_map,
             depth, q_len, num_qo_heads, num_kv_heads, num_splits, batch_size,
             scale_log2e);
        int8_merge_cuda(po, pl, o, HR, batch_size, num_splits, 256, stream);
    }
}
