#include "int8_decode.cuh"
#include <cstdio>

// One warp per CTA. Each lane owns HD/32 contiguous dims of one q-head; the warp
// cooperates on one key at a time (dot via shfl reduction), streaming its split's
// key range with online softmax. INT8 K/V dequantized in-register with per-(token,
// head) fp16 scales. No shared memory, no block syncs; 32-thread CTAs pack to full
// warp occupancy. Partials merged by int8_decode_merge_kernel.

template <int HD>
__global__ void __launch_bounds__(32, 32)
int8_decode_kernel
(
    const half* __restrict__ q,
    const int8_t* __restrict__ k_i8,
    const int8_t* __restrict__ v_i8,
    const half* __restrict__ k_scale,
    const half* __restrict__ v_scale,
    float* __restrict__ po,
    float* __restrict__ pl,
    const int* __restrict__ kv_len_dev,
    int depth,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float scale_log2e
)
{
    constexpr int DPL = HD / 32;
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    const int b = blockIdx.z;
    // Per-row kv_lens: scalar (1,) for single-row decode, (batch,) for batched rounds.
    const int kv_len = kv_len_dev[b];
    const int lane = threadIdx.x;
    const int d0 = lane * DPL;
    const int kvh = h / (num_qo_heads / num_kv_heads);

    const int chunk = (kv_len + num_splits - 1) / num_splits;
    const int kv_begin = split * chunk;
    int kv_end = kv_begin + chunk;
    if (kv_end > kv_len) kv_end = kv_len;

    // Q dims for this lane, pre-scaled for exp2f softmax.
    float qf[DPL];
    {
        const half* qp = q + ((long)b * num_qo_heads + h) * HD + d0;
        #pragma unroll
        for (int i = 0; i < DPL; i++) qf[i] = __half2float(qp[i]) * scale_log2e;
    }

    float oacc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; i++) oacc[i] = 0.0f;
    float m_i = -1e30f, l_i = 0.0f;

    const long kv_row = (long)depth * num_kv_heads;
    const int8_t* kbase = k_i8 + ((long)b * depth) * num_kv_heads * HD + (long)kvh * HD + d0;
    const int8_t* vbase = v_i8 + ((long)b * depth) * num_kv_heads * HD + (long)kvh * HD + d0;
    const half* skbase = k_scale + (long)b * depth * num_kv_heads + kvh;
    const half* svbase = v_scale + (long)b * depth * num_kv_heads + kvh;

    int8_t kr[DPL], vr[DPL];
    float sk = 0.0f, sv = 0.0f;
    auto load_key = [&](int key)
    {
        const int8_t* kp = kbase + (long)key * num_kv_heads * HD;
        const int8_t* vp = vbase + (long)key * num_kv_heads * HD;
        #pragma unroll
        for (int i = 0; i < DPL; i += 4)
        {
            *(int*)(kr + i) = *(const int*)(kp + i);
            *(int*)(vr + i) = *(const int*)(vp + i);
        }
        sk = __half2float(skbase[(long)key * num_kv_heads]);
        sv = __half2float(svbase[(long)key * num_kv_heads]);
    };

    if (kv_begin < kv_end) load_key(kv_begin);
    for (int key = kv_begin; key < kv_end; key++)
    {
        // prefetch next key's bytes while computing this one
        int8_t kr2[DPL], vr2[DPL];
        float sk2 = 0.0f, sv2 = 0.0f;
        if (key + 1 < kv_end)
        {
            const int8_t* kp = kbase + (long)(key + 1) * num_kv_heads * HD;
            const int8_t* vp = vbase + (long)(key + 1) * num_kv_heads * HD;
            #pragma unroll
            for (int i = 0; i < DPL; i += 4)
            {
                *(int*)(kr2 + i) = *(const int*)(kp + i);
                *(int*)(vr2 + i) = *(const int*)(vp + i);
            }
            sk2 = __half2float(skbase[(long)(key + 1) * num_kv_heads]);
            sv2 = __half2float(svbase[(long)(key + 1) * num_kv_heads]);
        }

        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL; i++) s = fmaf(qf[i], (float)kr[i] * sk, s);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);

        float mnew = fmaxf(m_i, s);
        float alpha = exp2f(m_i - mnew);
        float p = exp2f(s - mnew);
        l_i = l_i * alpha + p;
        #pragma unroll
        for (int i = 0; i < DPL; i++)
            oacc[i] = oacc[i] * alpha + p * ((float)vr[i] * sv);
        m_i = mnew;

        #pragma unroll
        for (int i = 0; i < DPL; i++) { kr[i] = kr2[i]; vr[i] = vr2[i]; }
        sk = sk2; sv = sv2;
    }

    long pbase = (((long)b * num_splits + split) * num_qo_heads + h) * HD;
    #pragma unroll
    for (int i = 0; i < DPL; i++) po[pbase + d0 + i] = oacc[i];
    if (lane == 0)
    {
        long mi = (((long)b * num_splits + split) * num_qo_heads + h) * 2;
        pl[mi + 0] = m_i;
        pl[mi + 1] = l_i;
    }
}

// Merge split partials: one block per (batch, q-head), HD threads stripe the dims.
template <int HD>
__global__ void int8_decode_merge_kernel
(
    const float* __restrict__ po,
    const float* __restrict__ pl,
    half* __restrict__ o,
    int num_qo_heads,
    int num_splits
)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d = threadIdx.x;
    __shared__ float s_w[2048];
    __shared__ float s_gm, s_den;

    if (d == 0)
    {
        float gm = -1e30f;
        for (int s = 0; s < num_splits; s++)
        {
            long mi = (((long)b * num_splits + s) * num_qo_heads + h) * 2;
            gm = fmaxf(gm, pl[mi + 0]);
        }
        s_gm = gm;
        s_den = 0.0f;
    }
    __syncthreads();
    const float gm = s_gm;
    for (int s = d; s < num_splits; s += HD)
    {
        long mi = (((long)b * num_splits + s) * num_qo_heads + h) * 2;
        float w = exp2f(pl[mi + 0] - gm);
        s_w[s] = w;
        atomicAdd(&s_den, w * pl[mi + 1]);
    }
    __syncthreads();
    const float inv = (s_den > 0.0f) ? 1.0f / s_den : 0.0f;
    float num = 0.0f;
    for (int s = 0; s < num_splits; s++)
    {
        long pi = (((long)b * num_splits + s) * num_qo_heads + h) * HD + d;
        num += s_w[s] * po[pi];
    }
    o[((long)b * num_qo_heads + h) * HD + d] = __float2half(num * inv);
}

void int8_decode_cuda
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
    int depth,
    int batch_size,
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
    dim3 grid(num_qo_heads, num_splits, batch_size);
    if (head_dim == 512)
    {
        int8_decode_kernel<512><<<grid, 32, 0, stream>>>
            (q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev,
             depth, num_qo_heads, num_kv_heads, num_splits, scale_log2e);
    }
    else if (head_dim == 256)
    {
        int8_decode_kernel<256><<<grid, 32, 0, stream>>>
            (q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev,
             depth, num_qo_heads, num_kv_heads, num_splits, scale_log2e);
    }
    else return;

    dim3 mgrid(num_qo_heads, batch_size);
    if (head_dim == 512)
        int8_decode_merge_kernel<512><<<mgrid, 512, 0, stream>>>(po, pl, o, num_qo_heads, num_splits);
    else
        int8_decode_merge_kernel<256><<<mgrid, 256, 0, stream>>>(po, pl, o, num_qo_heads, num_splits);
}
