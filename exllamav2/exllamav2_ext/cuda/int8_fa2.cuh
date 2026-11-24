#ifndef _int8_fa2_cuh
#define _int8_fa2_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FA2-style pipelined INT8 verify attention (hd512, small NQ). Declaration-only:
// this header is included by the HOST-compiled ext_cache.cpp, so the kernel and
// all mma/cp.async PTX live in int8_fa2.cu (same rule as int8_decode.cuh).
//
// Balanced-warp design: every warp does QK^T for its key sub-tile AND PV for its
// HD slice, so no warp idles during softmax (the HMMA kernel's QK ran on 1 of 4
// HD-split warps). K/V stream through double-buffered cp.async INT8 smem tiles and
// are dequantized to half in registers at fragment-load time -- no half smem tiles,
// which is what makes TK=32 fit alongside Q at hd512.
//
// Same split-KV po/pl contract as int8_attn_fast / int8_attn_hmma; merged by
// int8_merge_cuda. Graph-safe: device-side kv_len, caller-owned persistent
// partials, optional logical->physical row_map.
void int8_attn_fa2_cuda(
    const half* q, const int8_t* k_i8, const int8_t* v_i8,
    const half* k_scale, const half* v_scale, half* o,
    float* po, float* pl, const int* kv_len_dev, const int* row_map,
    int depth, int batch_size, int q_len, int num_qo_heads, int num_kv_heads,
    int num_splits, float sm_scale, int head_dim, cudaStream_t stream);

#endif
