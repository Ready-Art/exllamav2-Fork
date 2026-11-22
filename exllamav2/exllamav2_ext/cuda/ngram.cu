#include "ngram.cuh"

template <int bits>
__global__ void ngram_dequant_kernel
(
    const uint32_t* __restrict__ packed,
    const half* __restrict__ scales,
    const int32_t* __restrict__ heads,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int dim
)
{
    constexpr int vpw = 32 / bits;          // values per packed word
    constexpr int zero = 1 << (bits - 1);
    constexpr int mask = (1 << bits) - 1;

    int row = blockIdx.x;
    int t = threadIdx.x;
    int words = dim / vpw;
    if (t >= words) return;

    uint32_t w = packed[row * words + t];
    half s = scales[row];
    const half* b = bias + (int) heads[row] * dim;
    half* o = out + row * dim + t * vpw;

    #pragma unroll
    for (int i = 0; i < vpw; i++)
    {
        int q = (int) ((w >> (i * bits)) & mask);
        o[i] = __hadd(__hmul(__int2half_rn(q - zero), s), b[t * vpw + i]);
    }
}

void ngram_dequant_rows_cuda
(
    cudaStream_t stream,
    const uint32_t* packed,
    const half* scales,
    const int32_t* heads,
    const half* bias,
    half* out,
    int num_rows,
    int dim,
    int bits
)
{
    if (num_rows == 0) return;
    dim3 grid(num_rows), block(128);
    if (bits == 4)
        ngram_dequant_kernel<4><<<grid, block, 0, stream>>>(packed, scales, heads, bias, out, dim);
    else if (bits == 8)
        ngram_dequant_kernel<8><<<grid, block, 0, stream>>>(packed, scales, heads, bias, out, dim);
}
