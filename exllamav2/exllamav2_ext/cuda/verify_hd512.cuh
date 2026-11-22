#ifndef VERIFY_HD512_CUH
#define VERIFY_HD512_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Split-KV GQA-grouped causal attention for the MTP verify window (head_dim 512,
// small q_len, long kv) on sm86. The fa2 paged-prefill verify kernel reads the KV
// prefix once but runs far below bandwidth peak at q~3 (few query tiles -> low SM
// occupancy); per-row split-KV decode is bandwidth-optimal but reads the prefix
// q_len times (measured slower). This kernel is the missing middle: ONE pass over
// the KV (each key read once), parallelized across KV splits, with all q_len rows
// and the whole GQA group processed per CTA so the KV is not re-read per head.
//
// Two kernels: a partial-attention kernel (grid = num_splits x num_kv_heads, each
// CTA owns a KV chunk and computes unnormalized online-softmax partials for every
// query row / q-head) and a merge kernel (log-sum-exp combine of the split partials).
// Lower-right causal alignment (kv_len >= q_len): query row r attends keys
// <= kv_len - q_len + r. fp16 in/out, fp32 partials/accumulation, no window/softcap.
//
// Graph-safe: launched with a fixed split count and fixed workspace; kv_len is a
// kernel argument (refreshed per replay via the captured launch), buffers are
// caller-owned and never reallocated.

// Partial output layout: po[(split * q_len + r) * num_qo_heads + h] is a 512-vector
// of unnormalized exp2-weighted values; pl[(split * q_len + r) * num_qo_heads + h]
// holds (row_max_log2, row_sum). q is pre-scaled by sm_scale * log2(e).
void verify_hd512_cuda
(
    const half* q,          // (q_len, num_qo_heads, 512)
    const half* k,          // (kv_len, num_kv_heads, 512)
    const half* v,          // (kv_len, num_kv_heads, 512)
    half* o,                // (q_len, num_qo_heads, 512)
    float* po,              // (num_splits, q_len, num_qo_heads, 512) workspace
    float* pl,              // (num_splits, q_len, num_qo_heads, 2) workspace
    int q_len,
    const int* kv_len_dev,  // device scalar refreshed per replay (graph-safe kv_len)
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    float sm_scale,
    cudaStream_t stream
);

#endif