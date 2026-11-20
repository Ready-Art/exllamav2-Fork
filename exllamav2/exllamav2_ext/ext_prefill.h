#ifndef EXT_PREFILL_H
#define EXT_PREFILL_H

#include <torch/extension.h>
#include <cstdint>

void prefill_hd512
(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    double sm_scale
);

#endif