#ifndef _int8_attn_fast_cuh
#define _int8_attn_fast_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Fast INT8 KV attention: multi-warp CTAs + U=4 multi-key unroll (occupancy fix
// for the warp-per-CTA family). Serves decode (q_len=1) and verify (q_len>1,
// per-row causal frontier kv_len_dev[b] - q_len + r + 1). Graph-safe: device-side
// kv_len, caller-owned persistent partials (po, pl), optional row_map.
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
);

#endif
