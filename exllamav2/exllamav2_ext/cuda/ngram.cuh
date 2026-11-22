#ifndef _ngram_cuh
#define _ngram_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// EXL2.5 engram row dequant: symmetric uniform-bit rows + per-row fp16 scale + per-head
// bias. out[u][j] = (q[u][j] - 2^(bits-1)) * scale[u] + bias[head[u]][j].
// packed: (U, dim*bits/32) uint32, natural bit order (value j at shift j*bits).
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
);

#endif
