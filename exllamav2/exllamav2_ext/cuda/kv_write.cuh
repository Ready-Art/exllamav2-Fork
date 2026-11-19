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
    const int32_t* pos,      // device pointer, position of the first new token
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    int cache_len            // tokens per sequence in the cache buffer
);

#endif
