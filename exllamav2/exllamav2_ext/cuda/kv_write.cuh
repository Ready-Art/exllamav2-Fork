#ifndef _kv_write_cuh
#define _kv_write_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Copy the newly computed K/V rows into the KV cache at a position read from *device* memory.
//
// The ordinary path does this with a host-side slice
// (k_cache[:batch, past_len:past_len+q_len, :, :].copy_(k)), which bakes `past_len` into the
// kernel arguments and therefore cannot be captured into a CUDA graph: the graph would replay
// the position it was captured at. Reading the position from a device tensor instead means the
// same graph replays correctly at every step, as long as the tensor is updated before replay.
//
// Layout: cache is [batch, seq, kv_heads, head_dim] contiguous; src is [batch, q_len, kv_heads,
// head_dim] contiguous.
void kv_write_cuda
(
    cudaStream_t stream,
    const half* src_k,
    const half* src_v,
    half* cache_k,
    half* cache_v,
    const int32_t* pos,      // device pointer, per-sequence position of the first new token
    const int32_t* row_map,  // device pointer, cache row per sequence (NULL = identity)
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    int cache_len            // tokens per sequence in the cache buffer
);

// Quantize freshly written fp16 cache rows into an INT8 shadow + per-(token,head)
// fp16 scales (Ampere INT8 KV; same device-side position logic as kv_write_cuda).
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
);

// INT8-PRIMARY write: quantize the NEW K/V rows (src: batch, q_len, kvh, hd)
// straight into the full-depth INT8 cache at device-side pos[b]+t / row_map[b].
// No fp16 cache copy exists in this mode -- this replaces kv_write + quant_shadow
// for full-attention layers. Graph-safe (same position logic as kv_write_cuda).
void kv_quant_new_cuda
(
    cudaStream_t stream,
    const half* src_k,
    const half* src_v,
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
    int dst_depth
);

// Dequantize INT8 KV rows [start, start+n) of ONE cache row (src: depth, kvh, hd
// + scales depth, kvh) into a contiguous fp16 chunk (n, kvh, hd). Eager-only
// (prefill continuation staging / gather fallback); the read-side scale rounding
// matches kv_quant_* exactly so a quant->dequant round trip is the stored value.
void kv_dequant_cuda
(
    cudaStream_t stream,
    const int8_t* k_i8,
    const int8_t* v_i8,
    const half* k_scale,
    const half* v_scale,
    half* dst_k,
    half* dst_v,
    int start,
    int n,
    int kv_heads,
    int head_dim
);

// Quantize a contiguous fp16 chunk (n, kvh, hd) into one row of a full-depth INT8
// destination at absolute row dst_row (source/destination depths differ).
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
);

#endif
