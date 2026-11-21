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
