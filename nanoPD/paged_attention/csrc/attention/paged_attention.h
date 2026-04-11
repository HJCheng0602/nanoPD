#pragma once
#include<torch/extension.h>
#include "../utils/cuda_utils.h"



template <typename scalar_t>
__global__ void paged_attention_kernel(
    scalar_t * __restrict__ out,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ k_cache,
    const scalar_t * __restrict__ v_cache,
    const int * __restrict__ block_tables,
    const int * __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int head_size,
    const int num_kv_heads,
    const int block_size,
    const int max_num_blocks_per_seq
);


void paged_attention_forward(
    torch::Tensor& out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    float scale,
    int block_size,
    int max_num_blocks_per_seq
);
