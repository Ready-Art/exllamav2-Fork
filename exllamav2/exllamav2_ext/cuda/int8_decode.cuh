#ifndef _int8_decode_cuh
#define _int8_decode_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// INT8 KV flash-decode (q_len=1, GQA, contiguous KV, split-KV). Ampere has no FP8;
// KV is symmetric per-(token,head) INT8 with fp16 scales, dequantized in-register.
// One WARP per CTA (no block syncs, no shared): grid (nq, num_splits, batch), each
// warp streams its key sub-range with online softmax and writes unnormalized
// partials (po, pl=(m,l)) merged by int8_decode_merge. Graph-safe: device-side
// kv_len, fixed buffers. HD template: 512 (full layers) or 256 (SWA, later).

void int8_decode_cuda
(
    const half* q,            // (batch, nq, HD)
    const int8_t* k_i8,       // (batch, depth, nkv, HD)
    const int8_t* v_i8,
    const half* k_scale,      // (batch, depth, nkv)
    const half* v_scale,
    half* o,                  // (batch, nq, HD)
    float* po,                // (batch, num_splits, nq, HD)
    float* pl,                // (batch, num_splits, nq, 2)
    const int* kv_len_dev,    // device scalar or (batch,) per-row lengths
    const int* row_map,       // optional (batch,) logical->physical shadow row, NULL = identity
    int depth,                // cache depth (batch stride)
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float sm_scale,
    int head_dim,
    cudaStream_t stream
);

// INT8 verify attention (q_len rows, causal limit kv_len_dev[b]-q_len+r+1 per row).
// q (batch, q_len, nq, HD); partials po/pl sized (batch, splits, nq*q_len, ...).
void int8_verify_cuda
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
