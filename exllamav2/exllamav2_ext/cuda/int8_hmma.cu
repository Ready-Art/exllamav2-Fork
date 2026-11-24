#include "int8_hmma.cuh"
#include "int8_decode.cuh"

// Host launcher for the HMMA (tensor-core) INT8 verify attention kernel.
// Serves the MTP verify window (q_len > 1) on hd512 full-attention layers when the
// per-kv-head query count NQ = (nq/nkv)*q_len is 16 or 32 (Gemma-4 W=1/W=3). Decode
// (q_len=1, NQ<16) and other geometries stay on the warp int8_attn_fast kernel.
// Same po/pl split-KV contract as int8_attn_fast; merged by int8_merge_cuda.

template <int NQ>
static void launch_hmma(
    const half* q, const int8_t* k8, const int8_t* v8, const half* ks, const half* vs,
    float* po, float* pl, const int* kvl, const int* row_map,
    int depth, int qlen, int nq, int nkv, int splits, int B, float sc, cudaStream_t stream)
{
    constexpr int HD = 512, TK = 16, HSPLIT = 4;
    int smem = 2 * TK * HD * 2;
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(int8_verify_hmma_kernel<HD, TK, NQ, HSPLIT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    int total = nkv * splits * B;
    int8_verify_hmma_kernel<HD, TK, NQ, HSPLIT><<<total, (NQ / 16) * HSPLIT * 32, smem, stream>>>(
        q, k8, v8, ks, vs, po, pl, kvl, row_map, depth, qlen, nq, nkv, splits, B, sc);
}

void int8_attn_hmma_cuda(
    const half* q, const int8_t* k_i8, const int8_t* v_i8,
    const half* k_scale, const half* v_scale, half* o,
    float* po, float* pl, const int* kv_len_dev, const int* row_map,
    int depth, int batch_size, int q_len, int num_qo_heads, int num_kv_heads,
    int num_splits, float sm_scale, int head_dim, cudaStream_t stream)
{
    if (head_dim != 512 || num_kv_heads <= 0) return;
    int group = num_qo_heads / num_kv_heads;
    int nq_rt = group * q_len;
    float scale_log2e = sm_scale * 1.4426950408889634f;
    if (nq_rt == 16)
        launch_hmma<16>(q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev, row_map,
                        depth, q_len, num_qo_heads, num_kv_heads, num_splits, batch_size,
                        scale_log2e, stream);
    else if (nq_rt == 32)
        launch_hmma<32>(q, k_i8, v_i8, k_scale, v_scale, po, pl, kv_len_dev, row_map,
                        depth, q_len, num_qo_heads, num_kv_heads, num_splits, batch_size,
                        scale_log2e, stream);
    else
        return;   // unsupported geometry — caller falls back to the warp kernel
    int HR = num_qo_heads * q_len;
    int8_merge_cuda(po, pl, o, HR, batch_size, num_splits, 512, stream);
}
