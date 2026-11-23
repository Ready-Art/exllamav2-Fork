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
    const int* kv_len_dev,    // device scalar
    int depth,                // cache depth (batch stride)
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float sm_scale,
    int head_dim,
    cudaStream_t stream
);

#endif
