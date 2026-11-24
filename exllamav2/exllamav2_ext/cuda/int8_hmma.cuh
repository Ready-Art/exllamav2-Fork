#ifndef _int8_hmma_cuh
#define _int8_hmma_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// HMMA (tensor-core) INT8 verify attention — flash-attention tiled kernel, HD-split.
// QK^T and PV via nvcuda::wmma m16n16k16 half fragments; INT8 K/V dequantized to
// half on load. Split-KV across CTAs + the int8_decode_merge po/pl contract.
//
// The kernel itself lives in int8_hmma.cu: this header is included by the
// HOST-compiled ext_cache.cpp, so it must stay declaration-only (no <mma.h>,
// nvcuda or __global__ definitions -- same pattern as int8_decode.cuh).
//
// Host launcher. hd512 verify, NQ=(nq/nkv)*q_len in {16,32}; other geometries
// return without launching so the caller falls back to the warp kernel.
void int8_attn_hmma_cuda(
    const half* q, const int8_t* k_i8, const int8_t* v_i8,
    const half* k_scale, const half* v_scale, half* o,
    float* po, float* pl, const int* kv_len_dev, const int* row_map,
    int depth, int batch_size, int q_len, int num_qo_heads, int num_kv_heads,
    int num_splits, float sm_scale, int head_dim, cudaStream_t stream);

#endif
