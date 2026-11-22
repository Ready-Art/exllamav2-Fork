void ngram_dequant_rows
(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor heads,
    torch::Tensor bias,
    torch::Tensor out,
    int bits
);
