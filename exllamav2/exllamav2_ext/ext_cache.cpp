#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_cache.h"

#include "cuda/cache.cuh"
#include "cuda/kv_write.cuh"
#include "cuda/int8_decode.cuh"
#include "cuda/int8_attn_fast.cuh"
#include "cuda/int8_hmma.cuh"
#include "cuda/int8_fa2.cuh"
#include "cuda/mtp_int8.cuh"

#include "cpp/util.h"

// INT8 KV shadow quantization (Ampere: INT8 + fp16 per-(token,head) scales, no FP8).
void kv_quant_shadow
(
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor pos,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(cache_k, kHalf);
    TORCH_CHECK_DTYPE(cache_v, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(pos, kInt);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(cache_k));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int cache_len = (int) cache_k.size(1);
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    kv_quant_shadow_cuda
    (
        stream,
        (const half*) cache_k.data_ptr(),
        (const half*) cache_v.data_ptr(),
        (int8_t*) k_i8.data_ptr(),
        (int8_t*) v_i8.data_ptr(),
        (half*) k_scale.data_ptr(),
        (half*) v_scale.data_ptr(),
        (const int32_t*) pos.data_ptr(),
        row_map_ptr,
        batch_size, q_len, kv_heads, head_dim, cache_len
    );
}

// INT8 KV flash-decode (q_len=1, GQA, contiguous KV, split-KV, graph-safe).
void int8_decode
(
    torch::Tensor q,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    double sm_scale,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    TORCH_CHECK_DTYPE(po, kFloat);
    TORCH_CHECK_DTYPE(pl, kFloat);
    TORCH_CHECK_DTYPE(kv_len_dev, kInt);
    // Scalar (1,) for single-row decode, (batch_size,) for batched rounds -- the
    // kernel reads kv_len_dev[b] per row.
    TORCH_CHECK(kv_len_dev.numel() == 1 || kv_len_dev.numel() == batch_size,
                "int8_decode: kv_len_dev must be scalar or per-batch-row");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int depth = (int) k_i8.size(1);
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        TORCH_CHECK(row_map.numel() >= batch_size, "int8_decode: row_map too short");
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    int8_decode_cuda
    (
        (const half*) q.data_ptr(),
        (const int8_t*) k_i8.data_ptr(),
        (const int8_t*) v_i8.data_ptr(),
        (const half*) k_scale.data_ptr(),
        (const half*) v_scale.data_ptr(),
        (half*) o.data_ptr(),
        (float*) po.data_ptr(),
        (float*) pl.data_ptr(),
        (const int*) kv_len_dev.data_ptr(),
        row_map_ptr,
        depth, batch_size, num_qo_heads, num_kv_heads, num_splits,
        (float) sm_scale, head_dim, stream
    );
}

// INT8 verify attention (q_len rows over the INT8 KV, causal per-row limit).
// Graph-safe: device-side kv_len frontier, persistent partial buffers, optional row_map.
void int8_verify
(
    torch::Tensor q,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int batch_size,
    int q_len,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    double sm_scale,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    TORCH_CHECK_DTYPE(po, kFloat);
    TORCH_CHECK_DTYPE(pl, kFloat);
    TORCH_CHECK_DTYPE(kv_len_dev, kInt);
    TORCH_CHECK(kv_len_dev.numel() == 1 || kv_len_dev.numel() == batch_size,
                "int8_verify: kv_len_dev must be scalar or per-batch-row");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int depth = (int) k_i8.size(1);
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    int8_verify_cuda
    (
        (const half*) q.data_ptr(),
        (const int8_t*) k_i8.data_ptr(),
        (const int8_t*) v_i8.data_ptr(),
        (const half*) k_scale.data_ptr(),
        (const half*) v_scale.data_ptr(),
        (half*) o.data_ptr(),
        (float*) po.data_ptr(),
        (float*) pl.data_ptr(),
        (const int*) kv_len_dev.data_ptr(),
        row_map_ptr,
        depth, batch_size, q_len, num_qo_heads, num_kv_heads, num_splits,
        (float) sm_scale, head_dim, stream
    );
}

// Quantize a contiguous fp16 chunk (1, n, kvh, hd) into ONE shadow row at absolute
// row dst_row of a full-depth INT8 destination (MTP gather staging path).
void kv_quant_chunk
(
    torch::Tensor src_k,
    torch::Tensor src_v,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int dst_row,
    int n,
    int kv_heads,
    int head_dim
)
{
    TORCH_CHECK_DTYPE(src_k, kHalf);
    TORCH_CHECK_DTYPE(src_v, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(src_k));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    kv_quant_chunk_cuda
    (
        stream,
        (const half*) src_k.data_ptr(),
        (const half*) src_v.data_ptr(),
        (int8_t*) k_i8.data_ptr(),
        (int8_t*) v_i8.data_ptr(),
        (half*) k_scale.data_ptr(),
        (half*) v_scale.data_ptr(),
        dst_row, n, kv_heads, head_dim, (int) k_i8.size(1)
    );
}

// INT8-primary KV write: quantize the NEW projection rows straight into the
// full-depth INT8 cache (no fp16 cache copy). Graph-safe device pos/row_map.
void kv_quant_new
(
    torch::Tensor src_k,
    torch::Tensor src_v,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor pos,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(src_k, kHalf);
    TORCH_CHECK_DTYPE(src_v, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(pos, kInt);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(src_k));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    kv_quant_new_cuda
    (
        stream,
        (const half*) src_k.data_ptr(),
        (const half*) src_v.data_ptr(),
        (int8_t*) k_i8.data_ptr(),
        (int8_t*) v_i8.data_ptr(),
        (half*) k_scale.data_ptr(),
        (half*) v_scale.data_ptr(),
        (const int32_t*) pos.data_ptr(),
        row_map_ptr,
        batch_size, q_len, kv_heads, head_dim, (int) k_i8.size(1)
    );
}

// INT8 -> fp16 dequantize of one cache row's [start, start+n) prefix into a
// contiguous fp16 chunk (prefill continuation staging / gather fallback).
void kv_dequant
(
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor dst_k,
    torch::Tensor dst_v,
    int start,
    int n,
    int kv_heads,
    int head_dim
)
{
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(dst_k, kHalf);
    TORCH_CHECK_DTYPE(dst_v, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(k_i8));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    kv_dequant_cuda
    (
        stream,
        (const int8_t*) k_i8.data_ptr(),
        (const int8_t*) v_i8.data_ptr(),
        (const half*) k_scale.data_ptr(),
        (const half*) v_scale.data_ptr(),
        (half*) dst_k.data_ptr(),
        (half*) dst_v.data_ptr(),
        start, n, kv_heads, head_dim
    );
}

// Fast INT8 attention (multi-warp CTAs + multi-key unroll): decode q_len=1 and
// verify q_len>1 share one kernel. Same buffer contract as int8_decode/verify.
void int8_attn_fast
(
    torch::Tensor q,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int batch_size,
    int q_len,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    double sm_scale,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    TORCH_CHECK_DTYPE(po, kFloat);
    TORCH_CHECK_DTYPE(pl, kFloat);
    TORCH_CHECK_DTYPE(kv_len_dev, kInt);
    TORCH_CHECK(kv_len_dev.numel() == 1 || kv_len_dev.numel() == batch_size,
                "int8_attn_fast: kv_len_dev must be scalar or per-batch-row");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int depth = (int) k_i8.size(1);
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        TORCH_CHECK(row_map.numel() >= batch_size, "int8_attn_fast: row_map too short");
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    int8_attn_fast_cuda
    (
        (const half*) q.data_ptr(),
        (const int8_t*) k_i8.data_ptr(),
        (const int8_t*) v_i8.data_ptr(),
        (const half*) k_scale.data_ptr(),
        (const half*) v_scale.data_ptr(),
        (half*) o.data_ptr(),
        (float*) po.data_ptr(),
        (float*) pl.data_ptr(),
        (const int*) kv_len_dev.data_ptr(),
        row_map_ptr,
        depth, batch_size, q_len, num_qo_heads, num_kv_heads, num_splits,
        (float) sm_scale, head_dim, stream
    );
}

// HMMA (tensor-core) INT8 verify attention — same buffer contract as int8_attn_fast.
// Serves hd512 full-layer verify windows with NQ=(nq/nkv)*q_len in {16,32}; the
// FA2-style pipelined INT8 verify attention (balanced-warp mma + cp.async INT8 tiles).
// Same contract as int8_attn_hmma; launcher no-ops on unsupported geometries.
void int8_attn_fa2
(
    torch::Tensor q,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int batch_size,
    int q_len,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    double sm_scale,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    TORCH_CHECK_DTYPE(po, kFloat);
    TORCH_CHECK_DTYPE(pl, kFloat);
    TORCH_CHECK_DTYPE(kv_len_dev, kInt);
    TORCH_CHECK(kv_len_dev.numel() == 1 || kv_len_dev.numel() == batch_size,
                "int8_attn_fa2: kv_len_dev must be scalar or per-batch-row");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int depth = (int) k_i8.size(1);
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        TORCH_CHECK(row_map.numel() >= batch_size, "int8_attn_fa2: row_map too short");
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    int8_attn_fa2_cuda
    (
        (const half*) q.data_ptr(),
        (const int8_t*) k_i8.data_ptr(),
        (const int8_t*) v_i8.data_ptr(),
        (const half*) k_scale.data_ptr(),
        (const half*) v_scale.data_ptr(),
        (half*) o.data_ptr(),
        (float*) po.data_ptr(),
        (float*) pl.data_ptr(),
        (const int*) kv_len_dev.data_ptr(),
        row_map_ptr,
        depth, batch_size, q_len, num_qo_heads, num_kv_heads, num_splits,
        (float) sm_scale, head_dim, stream
    );
}

// launcher no-ops on other geometries (caller falls back to int8_attn_fast).
void int8_attn_hmma
(
    torch::Tensor q,
    torch::Tensor k_i8,
    torch::Tensor v_i8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int batch_size,
    int q_len,
    int num_qo_heads,
    int num_kv_heads,
    int num_splits,
    double sm_scale,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k_i8, kChar);
    TORCH_CHECK_DTYPE(v_i8, kChar);
    TORCH_CHECK_DTYPE(k_scale, kHalf);
    TORCH_CHECK_DTYPE(v_scale, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    TORCH_CHECK_DTYPE(po, kFloat);
    TORCH_CHECK_DTYPE(pl, kFloat);
    TORCH_CHECK_DTYPE(kv_len_dev, kInt);
    TORCH_CHECK(kv_len_dev.numel() == 1 || kv_len_dev.numel() == batch_size,
                "int8_attn_hmma: kv_len_dev must be scalar or per-batch-row");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int depth = (int) k_i8.size(1);
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        TORCH_CHECK(row_map.numel() >= batch_size, "int8_attn_hmma: row_map too short");
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }
    int8_attn_hmma_cuda
    (
        (const half*) q.data_ptr(),
        (const int8_t*) k_i8.data_ptr(),
        (const int8_t*) v_i8.data_ptr(),
        (const half*) k_scale.data_ptr(),
        (const half*) v_scale.data_ptr(),
        (half*) o.data_ptr(),
        (float*) po.data_ptr(),
        (float*) pl.data_ptr(),
        (const int*) kv_len_dev.data_ptr(),
        row_map_ptr,
        depth, batch_size, q_len, num_qo_heads, num_kv_heads, num_splits,
        (float) sm_scale, head_dim, stream
    );
}

// Weight-only INT8 GEMV for the MTP drafter: out(m,N) = x(m,K) fp16 @ w8(N,K)^T *
// scale(N) fp16. m <= 8. Output allocated here (inside a graph capture it lands in
// the capture pool, stable across replays -- same contract as torch matmul).
torch::Tensor mtp_int8_gemm
(
    torch::Tensor x,
    torch::Tensor w8,
    torch::Tensor scale
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w8, kChar);
    TORCH_CHECK_DTYPE(scale, kHalf);
    TORCH_CHECK(x.dim() == 2 && w8.dim() == 2, "mtp_int8_gemm: x (m,K), w8 (N,K)");
    TORCH_CHECK(x.size(1) == w8.size(1), "mtp_int8_gemm: K mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8, "mtp_int8_gemm: m must be 1..8");
    TORCH_CHECK(scale.numel() == w8.size(0), "mtp_int8_gemm: scale rows mismatch");
    TORCH_CHECK(x.is_contiguous() && w8.is_contiguous() && scale.is_contiguous(),
                "mtp_int8_gemm: contiguous inputs required");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int m = (int) x.size(0);
    int N = (int) w8.size(0);
    int K = (int) w8.size(1);
    torch::Tensor out = torch::empty({m, N}, x.options());
    mtp_int8_gemm_cuda
    (
        (const half*) x.data_ptr(),
        (const int8_t*) w8.data_ptr(),
        (const half*) scale.data_ptr(),
        (half*) out.data_ptr(),
        m, N, K, stream
    );
    return out;
}

// CUDA-graph-safe KV cache write. The position comes from a device tensor, so the same
// captured graph replays correctly at every decode step.
void kv_write
(
    torch::Tensor src_k,
    torch::Tensor src_v,
    torch::Tensor cache_k,
    torch::Tensor cache_v,
    torch::Tensor pos,
    int batch_size,
    int q_len,
    int kv_heads,
    int head_dim,
    torch::Tensor row_map
)
{
    TORCH_CHECK_DTYPE(src_k, kHalf);
    TORCH_CHECK_DTYPE(src_v, kHalf);
    TORCH_CHECK_DTYPE(cache_k, kHalf);
    TORCH_CHECK_DTYPE(cache_v, kHalf);
    TORCH_CHECK_DTYPE(pos, kInt);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(cache_k));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // cache is [batch, cache_len, kv_heads, head_dim]
    int cache_len = (int) cache_k.size(1);

    // Optional per-sequence cache-row mapping (multi-batch slot graphs). An undefined
    // or empty tensor means identity (sequence b writes cache row b), the original path.
    const int32_t* row_map_ptr = NULL;
    if (row_map.defined() && !row_map.device().is_meta() && row_map.numel() > 0)
    {
        TORCH_CHECK_DTYPE(row_map, kInt);
        row_map_ptr = (const int32_t*) row_map.data_ptr();
    }

    kv_write_cuda
    (
        stream,
        (const half*) src_k.data_ptr(),
        (const half*) src_v.data_ptr(),
        (half*) cache_k.data_ptr(),
        (half*) cache_v.data_ptr(),
        (const int32_t*) pos.data_ptr(),
        row_map_ptr,
        batch_size,
        q_len,
        kv_heads,
        head_dim,
        cache_len
    );
}

void fp16_to_fp8(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width)
{
    TORCH_CHECK_DTYPE(in_tensor, kHalf);
    TORCH_CHECK_DTYPE(out_tensor, kUInt8);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_tensor));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(in_tensor, 0, out_tensor, 0, 1);
    TORCH_CHECK_SHAPES(in_tensor, 1, out_tensor, 1, 1);
    TORCH_CHECK_SHAPES(in_tensor, 2, out_tensor, 2, 1);
    TORCH_CHECK_SHAPES(in_tensor, 3, out_tensor, 3, 1);

    int stride = in_tensor.size(1) * in_tensor.size(2) * in_tensor.size(3);
    int height = batch_size;

    int tsize = in_tensor.size(2) * in_tensor.size(3);
    offset *= tsize;
    width *= tsize;

    array_fp16_to_fp8_cuda
    (
        stream,
        (const half*) in_tensor.data_ptr(),
        (unsigned char*) out_tensor.data_ptr(),
        stride,
        height,
        offset,
        width
    );
}

void fp8_to_fp16(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width)
{
    TORCH_CHECK_DTYPE(in_tensor, kUInt8);
    TORCH_CHECK_DTYPE(out_tensor, kHalf);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_tensor));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(in_tensor, 0, out_tensor, 0, 1);
    TORCH_CHECK_SHAPES(in_tensor, 1, out_tensor, 1, 1);
    TORCH_CHECK_SHAPES(in_tensor, 2, out_tensor, 2, 1);
    TORCH_CHECK_SHAPES(in_tensor, 3, out_tensor, 3, 1);

    int stride = in_tensor.size(1) * in_tensor.size(2) * in_tensor.size(3);
    int height = batch_size;

    int tsize = in_tensor.size(2) * in_tensor.size(3);
    offset *= tsize;
    width *= tsize;

    array_fp8_to_fp16_cuda
    (
        stream,
        (const unsigned char*) in_tensor.data_ptr(),
        (half*) out_tensor.data_ptr(),
        stride,
        height,
        offset,
        width
    );
}

void fp16_to_q_kv
(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    int batch_size,
    int offset,
    int width,
    int page_size,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int wbits
)
{
    TORCH_CHECK_DTYPE(k_in, kHalf);
    TORCH_CHECK_DTYPE(k_out, kUInt8);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(k_in));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(k_in, 0, k_out, 0, 1);
    TORCH_CHECK_SHAPES(k_in, 1, k_out, 1, 1);
    TORCH_CHECK_SHAPES(k_in, 2, k_out, 2, 1);
//    TORCH_CHECK_SHAPES(k_in, 3, k_out, 3, 2);
    TORCH_CHECK_SHAPES(v_in, 0, v_out, 0, 1);
    TORCH_CHECK_SHAPES(v_in, 1, v_out, 1, 1);
    TORCH_CHECK_SHAPES(v_in, 2, v_out, 2, 1);
//    TORCH_CHECK_SHAPES(v_in, 3, v_out, 3, 2);
    TORCH_CHECK_SHAPES(k_in, 0, v_in, 0, 1);
    TORCH_CHECK_SHAPES(k_in, 1, v_in, 1, 1);
    TORCH_CHECK_SHAPES(k_in, 2, v_in, 2, 1);
//    TORCH_CHECK_SHAPES(k_in, 3, v_in, 3, 1);

    if (page_size)
    {
        int dim = k_in.size(2) * k_in.size(3);
        batch_size = block_table.size(0);
        int pages_per_seq = block_table.size(1);

        TORCH_CHECK_SHAPES(cache_seqlens, 0, block_table, 0, 1);
//        TORCH_CHECK(dim % 256 == 0, "(num_kv_heads * head_dim) must be divisible by 256");

        array_fp16_to_q_kv_paged_cuda
        (
            stream,
            (const half*) k_in.data_ptr(),
            (unsigned char*) k_out.data_ptr(),
            (half*) k_scales.data_ptr(),
            (const half*) v_in.data_ptr(),
            (unsigned char*) v_out.data_ptr(),
            (half*) v_scales.data_ptr(),
            batch_size,
            dim,
            pages_per_seq,
            (const int*) cache_seqlens.data_ptr(),
            (const int*) block_table.data_ptr(),
            page_size,
            width,
            wbits
        );
    }
    else
    {
        int stride = k_in.size(1) * k_in.size(2) * k_in.size(3);
        int height = batch_size;

        int dim = k_in.size(2) * k_in.size(3);
        if (dim % Q_CACHE_BLOCKSIZE_Q)
        {
            while ((offset * dim) % Q_CACHE_BLOCKSIZE_Q) offset--;
            while ((width * dim) % Q_CACHE_BLOCKSIZE_Q) width++;
        }
        offset *= dim;
        width *= dim;

        array_fp16_to_q_kv_cuda
        (
            stream,
            (const half*) k_in.data_ptr(),
            (unsigned char*) k_out.data_ptr(),
            (half*) k_scales.data_ptr(),
            (const half*) v_in.data_ptr(),
            (unsigned char*) v_out.data_ptr(),
            (half*) v_scales.data_ptr(),
            dim,
            stride,
            height,
            offset,
            width,
            wbits
        );
    }
}

void q_to_fp16_kv
(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    int batch_size,
    int offset,
    int width,
    int page_size,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int wbits
)
{
    TORCH_CHECK_DTYPE(k_in, kUInt8);
    TORCH_CHECK_DTYPE(k_out, kHalf);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(k_in));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(k_out, 0, k_in, 0, 1);
    TORCH_CHECK_SHAPES(k_out, 1, k_in, 1, 1);
    TORCH_CHECK_SHAPES(k_out, 2, k_in, 2, 1);
//    TORCH_CHECK_SHAPES(k_out, 3, k_in, 3, 2);
    TORCH_CHECK_SHAPES(v_out, 0, v_in, 0, 1);
    TORCH_CHECK_SHAPES(v_out, 1, v_in, 1, 1);
    TORCH_CHECK_SHAPES(v_out, 2, v_in, 2, 1);
//    TORCH_CHECK_SHAPES(v_out, 3, v_in, 3, 2);
    TORCH_CHECK_SHAPES(k_in, 0, v_in, 0, 1);
    TORCH_CHECK_SHAPES(k_in, 1, v_in, 1, 1);
    TORCH_CHECK_SHAPES(k_in, 2, v_in, 2, 1);
//    TORCH_CHECK_SHAPES(k_in, 3, v_in, 3, 1);

    if (page_size)
    {
        int dim = k_out.size(2) * k_out.size(3);
        batch_size = block_table.size(0);
        int pages_per_seq = block_table.size(1);

        TORCH_CHECK_SHAPES(cache_seqlens, 0, block_table, 0, 1);
//        TORCH_CHECK(dim % 256 == 0, "(num_kv_heads * head_dim) must be divisible by 256");

        array_q_to_fp16_kv_paged_cuda
        (
            stream,
            (const unsigned char*) k_in.data_ptr(),
            (const half*) k_scales.data_ptr(),
            (half*) k_out.data_ptr(),
            (const unsigned char*) v_in.data_ptr(),
            (const half*) v_scales.data_ptr(),
            (half*) v_out.data_ptr(),
            batch_size,
            dim,
            pages_per_seq,
            (const int*) cache_seqlens.data_ptr(),
            (const int*) block_table.data_ptr(),
            page_size,
            wbits
        );
    }
    else
    {
        int stride = k_out.size(1) * k_out.size(2) * k_out.size(3);
        int height = batch_size;

        int dim = k_out.size(2) * k_out.size(3);
        if (dim % Q_CACHE_BLOCKSIZE_Q)
        {
            while ((offset * dim) % Q_CACHE_BLOCKSIZE_Q) offset--;
            while ((width * dim) % Q_CACHE_BLOCKSIZE_Q) width++;
        }
        offset *= dim;
        width *= dim;

        array_q_to_fp16_kv_cuda
        (
            stream,
            (const unsigned char*) k_in.data_ptr(),
            (const half*) k_scales.data_ptr(),
            (half*) k_out.data_ptr(),
            (const unsigned char*) v_in.data_ptr(),
            (const half*) v_scales.data_ptr(),
            (half*) v_out.data_ptr(),
            dim,
            stride,
            height,
            offset,
            width,
            wbits
        );
    }
}

//void array_fp16_to_fp8_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size)
//{
//    TORCH_CHECK_DTYPE(in_tensor, kHalf);
//    TORCH_CHECK_DTYPE(out_tensor, kUInt8);
//    array_fp16_to_fp8_ref_cuda(NULL, (const half*) (in_tensor.data_ptr()), (unsigned char*)(out_tensor.data_ptr()), size);
//}
//
//void array_fp8_to_fp16_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size)
//{
//    TORCH_CHECK_DTYPE(in_tensor, kUInt8);
//    TORCH_CHECK_DTYPE(out_tensor, kHalf);
//    array_fp8_to_fp16_ref_cuda(NULL, (const unsigned char*)(in_tensor.data_ptr()), (half*)(out_tensor.data_ptr()), size);
//}

int count_match
(
    torch::Tensor a,
    torch::Tensor b,
    int max_a
)
{
    uint64_t* pa = (uint64_t*) a.data_ptr();
    uint64_t* pb = (uint64_t*) b.data_ptr();
    int max_b = b.size(1);
    if (max_b < max_a) max_a = max_b;

    int match = 0;
    while (match < max_a && *pa++ == *pb++)
        match++;

    return match;
}