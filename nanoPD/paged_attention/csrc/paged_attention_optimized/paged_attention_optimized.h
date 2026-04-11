#pragma once
#include<torch/extension.h>
#include "../utils/cuda_utils.h"


template <typename scalar_t, int HEAD_SIZE, int WARP_SIZE, int NUM_THREADS, int PARTITION_SIZE, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    scalar_t * __restrict__ partial_out,
    float * exp_sums,
    float * max_logits,
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
    const int max_num_blocks_per_seq,
    const int num_partition
);


template<typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__global__ void paged_attention_reduce_kernel(
    scalar_t* out,                    // [num_seqs, num_heads, head_size]
    const scalar_t* partial_out,      // [num_seqs, num_heads, num_partitions, head_size]
    const float* exp_sums,            // [num_seqs, num_heads, num_partitions]
    const float* max_logits,          // [num_seqs, num_heads, num_partitions]
    int num_heads, int num_partitions, int head_size);

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
