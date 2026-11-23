#include "kv_write.cuh"

// One thread per half2 element. Total elements = batch * q_len * kv_heads * head_dim / 2.
//
// The destination row is (pos + row_within_seq), read from device memory so the kernel is
// position-agnostic and can be replayed from a captured graph at any step.
__global__ void kv_write_kernel
(
    const half* __restrict__ src_k,
    const half* __restrict__ src_v,
    half* __restrict__ cache_k,
    half* __restrict__ cache_v,
    const int32_t* __restrict__ pos,
    const int32_t* __restrict__ row_map,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    int cache_len,
    int total_half2
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_half2) return;

    int row_elems = kv_heads * head_dim;          // elements per token
    int row_half2 = row_elems / 2;                // half2 per token

    int row = idx / row_half2;                    // global row in src, 0..batch*q_len-1
    int col = idx - row * row_half2;              // half2 index within the token

    int b = row / q_len;                          // which sequence
    int t = row - b * q_len;                      // token within this call

    // Multi-batch: each sequence has its own write position (pos[b]) and may live in
    // any cache row (row_map[b]); the batched graph replays for any active slot subset.
    // Single-sequence callers pass pos filled with one scalar and no row_map, which
    // reduces to the original pos[0] / row b behaviour exactly.
    int p = pos[b];
    if (p < 0) return;
    int cache_b = row_map ? row_map[b] : b;

    int dst_row = p + t;
    if (dst_row >= cache_len) return;             // never write past the buffer

    // Cache is [batch, cache_len, kv_heads, head_dim]; source is [batch, q_len, ...]
    size_t src_off = ((size_t) row * row_half2 + col);
    size_t dst_off = (((size_t) cache_b * cache_len + dst_row) * row_half2 + col);

    ((half2*) cache_k)[dst_off] = ((const half2*) src_k)[src_off];
    ((half2*) cache_v)[dst_off] = ((const half2*) src_v)[src_off];
}

void kv_write_cuda
(
    cudaStream_t stream,
    const half* src_k,
    const half* src_v,
    half* cache_k,
    half* cache_v,
    const int32_t* pos,
    const int32_t* row_map,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    int cache_len
)
{
    int total_half2 = batch_size * q_len * kv_heads * head_dim / 2;
    if (total_half2 <= 0) return;

    int threads = 256;
    int blocks = (total_half2 + threads - 1) / threads;

    kv_write_kernel<<<blocks, threads, 0, stream>>>
    (
        src_k,
        src_v,
        cache_k,
        cache_v,
        pos,
        row_map,
        batch_size,
        q_len,
        kv_heads,
        head_dim,
        cache_len,
        total_half2
    );
}

// Quantize freshly written fp16 cache rows into an INT8 shadow (Ampere has no FP8).
// Symmetric per-(token, head) scales: scale = max|row| / 127, stored fp16; the kernel
// dequantizes with the SAME fp16-rounded scale. One block per (row, head); position
// logic mirrors kv_write_kernel (device-side pos/row_map -> graph-safe).
__global__ void kv_quant_shadow_kernel
(
    const half* __restrict__ cache_k,
    const half* __restrict__ cache_v,
    int8_t* __restrict__ k_i8,
    int8_t* __restrict__ v_i8,
    half* __restrict__ k_scale,
    half* __restrict__ v_scale,
    const int32_t* __restrict__ pos,
    const int32_t* __restrict__ row_map,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    int cache_len
)
{
    int idx = blockIdx.x;
    int head = idx % kv_heads;
    int row = idx / kv_heads;
    int b = row / q_len;
    int t = row - b * q_len;

    int p = pos[b];
    if (p < 0) return;
    int cache_b = row_map ? row_map[b] : b;
    int dst_row = p + t;
    if (dst_row >= cache_len) return;

    long off = (((long)cache_b * cache_len + dst_row) * kv_heads + head) * head_dim;
    const half* kp = cache_k + off;
    const half* vp = cache_v + off;

    float mk = 0.0f, mv = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        mk = fmaxf(mk, fabsf(__half2float(kp[i])));
        mv = fmaxf(mv, fabsf(__half2float(vp[i])));
    }
    __shared__ float red[512];
    red[threadIdx.x] = mk;
    red[blockDim.x + threadIdx.x] = mv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            red[blockDim.x + threadIdx.x] = fmaxf(red[blockDim.x + threadIdx.x], red[blockDim.x + threadIdx.x + s]);
        }
        __syncthreads();
    }
    // Round the scale to fp16 FIRST so quantization and the read-side dequant agree.
    half shk = __float2half(red[0] / 127.0f);
    half shv = __float2half(red[blockDim.x] / 127.0f);
    float usk = __half2float(shk); if (usk == 0.0f) usk = 1e-8f;
    float usv = __half2float(shv); if (usv == 0.0f) usv = 1e-8f;

    long soff = ((long)cache_b * cache_len + dst_row) * kv_heads + head;
    if (threadIdx.x == 0)
    {
        k_scale[soff] = shk;
        v_scale[soff] = shv;
    }
    float inv_k = 1.0f / usk, inv_v = 1.0f / usv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        int qk = (int)rintf(__half2float(kp[i]) * inv_k);
        int qv = (int)rintf(__half2float(vp[i]) * inv_v);
        qk = max(-127, min(127, qk));
        qv = max(-127, min(127, qv));
        k_i8[off + i] = (int8_t)qk;
        v_i8[off + i] = (int8_t)qv;
    }
}

void kv_quant_shadow_cuda
(
    cudaStream_t stream,
    const half* cache_k,
    const half* cache_v,
    int8_t* k_i8,
    int8_t* v_i8,
    half* k_scale,
    half* v_scale,
    const int32_t* pos,
    const int32_t* row_map,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    int cache_len
)
{
    int blocks = batch_size * q_len * kv_heads;
    if (blocks <= 0) return;
    kv_quant_shadow_kernel<<<blocks, 256, 0, stream>>>
    (
        cache_k, cache_v, k_i8, v_i8, k_scale, v_scale,
        pos, row_map, batch_size, q_len, kv_heads, head_dim, cache_len
    );
}

// Quantize a contiguous fp16 chunk (n, kvh, hd) into ONE row of a full-depth INT8
// destination at absolute row dst_row (source and destination depths differ -- the
// MTP gather stages deltas through a small fp16 buffer when INT8 is primary).
__global__ void kv_quant_chunk_kernel
(
    const half* __restrict__ src_k,
    const half* __restrict__ src_v,
    int8_t* __restrict__ k_i8,
    int8_t* __restrict__ v_i8,
    half* __restrict__ k_scale,
    half* __restrict__ v_scale,
    int dst_row,
    int n,
    int kv_heads,
    int head_dim,
    int dst_depth
)
{
    int idx = blockIdx.x;
    int head = idx % kv_heads;
    int t = idx / kv_heads;
    if (t >= n) return;

    long soff = ((long)t * kv_heads + head) * head_dim;
    const half* kp = src_k + soff;
    const half* vp = src_v + soff;

    float mk = 0.0f, mv = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        mk = fmaxf(mk, fabsf(__half2float(kp[i])));
        mv = fmaxf(mv, fabsf(__half2float(vp[i])));
    }
    __shared__ float red[512];
    red[threadIdx.x] = mk;
    red[blockDim.x + threadIdx.x] = mv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            red[blockDim.x + threadIdx.x] = fmaxf(red[blockDim.x + threadIdx.x], red[blockDim.x + threadIdx.x + s]);
        }
        __syncthreads();
    }
    half shk = __float2half(red[0] / 127.0f);
    half shv = __float2half(red[blockDim.x] / 127.0f);
    float usk = __half2float(shk); if (usk == 0.0f) usk = 1e-8f;
    float usv = __half2float(shv); if (usv == 0.0f) usv = 1e-8f;

    long doff = (((long)dst_row + t) * kv_heads + head) * head_dim;
    long ss = ((long)dst_row + t) * kv_heads + head;
    if (threadIdx.x == 0)
    {
        k_scale[ss] = shk;
        v_scale[ss] = shv;
    }
    float inv_k = 1.0f / usk, inv_v = 1.0f / usv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        int qk = (int)rintf(__half2float(kp[i]) * inv_k);
        int qv = (int)rintf(__half2float(vp[i]) * inv_v);
        qk = max(-127, min(127, qk));
        qv = max(-127, min(127, qv));
        k_i8[doff + i] = (int8_t)qk;
        v_i8[doff + i] = (int8_t)qv;
    }
}

void kv_quant_chunk_cuda
(
    cudaStream_t stream,
    const half* src_k,
    const half* src_v,
    int8_t* k_i8,
    int8_t* v_i8,
    half* k_scale,
    half* v_scale,
    int dst_row,
    int n,
    int kv_heads,
    int head_dim,
    int dst_depth
)
{
    int blocks = n * kv_heads;
    if (blocks <= 0) return;
    kv_quant_chunk_kernel<<<blocks, 256, 0, stream>>>
    (
        src_k, src_v, k_i8, v_i8, k_scale, v_scale,
        dst_row, n, kv_heads, head_dim, dst_depth
    );
}
