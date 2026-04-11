#pragma once
#include<torch/extension.h>
#include"../utils/cuda_utils.h"



template <typename scalar_t>
__global__ void paged_kv_store_kernel(
    scalar_t * k_cache, scalar_t * v_cache,
    const scalar_t * k_src, const scalar_t * v_src,
    const int * __restrict__ block_tables,   // (num_tokens, max_blocks_per_seq)
    const int * __restrict__ positions,       // (num_tokens,)
    const int num_tokens,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq
);




void paged_kv_store(
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    const torch::Tensor& k_src, const torch::Tensor& v_src,
    const torch::Tensor& block_tables, const torch::Tensor& positions
);