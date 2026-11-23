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
    if (row_map.defined() && row_map.numel() > 0)
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
    int head_dim
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
    TORCH_CHECK(kv_len_dev.numel() == 1, "int8_decode: kv_len_dev must be a scalar");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int depth = (int) k_i8.size(1);
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
        depth, batch_size, num_qo_heads, num_kv_heads, num_splits,
        (float) sm_scale, head_dim, stream
    );
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
    if (row_map.defined() && row_map.numel() > 0)
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