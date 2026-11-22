#ifndef EXT_VERIFY_H
#define EXT_VERIFY_H

#include <torch/extension.h>

void verify_hd512
(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor po,
    torch::Tensor pl,
    torch::Tensor kv_len_dev,
    int64_t q_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t num_splits,
    double sm_scale
);

#endif