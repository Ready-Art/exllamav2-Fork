#ifndef _exl25_transform_cuh
#define _exl25_transform_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

class QMatrix;

int exl25_serve_kn(const QMatrix* b, int size_m, int size_n, int size_k);

void exl25_gemm_cuda_part
(
    cudaStream_t stream,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear,
    const half* r_weights,
    int r_weights_stride,
    bool mul_r_weights,
    class Graph* graph,
    int label,
    int kn,
    bool pre_tf = false
);

half* exl25_get_scratch(int device, size_t numel);

void exl25_transform_a
(
    cudaStream_t stream,
    const half* a,
    half* out,
    int size_m,
    int size_k,
    const half* awq_scale,
    int had_k
);

#endif
