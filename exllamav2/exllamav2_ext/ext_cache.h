
void fp16_to_fp8(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);

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
);
void fp8_to_fp16(torch::Tensor in_tensor, torch::Tensor out_tensor, int batch_size, int offset, int width);

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
);

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
);

int count_match
(
    torch::Tensor a,
    torch::Tensor b,
    int max_a
);

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
);

torch::Tensor mtp_int8_gemm
(
    torch::Tensor x,
    torch::Tensor w8,
    torch::Tensor scale
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

//void array_fp16_to_fp8_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size);
//void array_fp8_to_fp16_ref(torch::Tensor in_tensor, torch::Tensor out_tensor, int size);