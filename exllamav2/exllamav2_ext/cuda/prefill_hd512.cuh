#ifndef PREFILL_HD512_CUH
#define PREFILL_HD512_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused causal GQA prefill attention for head_dim 512 on sm86 (RTX 3090).
// FA2's hd512 kernel is register-starved (~9 TFLOPS measured on this box); this
// kernel tiles 16 query rows x 64 keys with wmma 16x16x16 and fp32 online softmax.
// Lower-right causal alignment (kv_len >= q_len), fp16 in/out, no window/softcap.
void prefill_hd512_cuda
(
    const half* q,          // (q_len, num_qo_heads, 512)
    const half* k,          // (kv_len, num_kv_heads, 512)
    const half* v,          // (kv_len, num_kv_heads, 512)
    half* o,                // (q_len, num_qo_heads, 512)
    int q_len,
    int kv_len,
    int num_qo_heads,
    int num_kv_heads,
    float sm_scale,
    cudaStream_t stream
);

#endif