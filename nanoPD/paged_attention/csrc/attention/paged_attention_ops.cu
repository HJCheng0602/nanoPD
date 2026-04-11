#include "paged_attention.h"
#include "../utils/cuda_utils.h"
#include "../utils/type_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include "paged_attention_kernel.cu"

void paged_attention_forward(
    torch::Tensor& out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    float scale,
    int block_size,
    int max_num_blocks_per_seq)
{
    TORCH_CHECK(query.is_cuda(), "query is must be on CUDA");
    TORCH_CHECK(query.dim() == 3, "query must be [num seqs, num_heads, head_size]")
    TORCH_CHECK(key_cache.dim() == 4)
    TORCH_CHECK(block_tables.dtype() == torch::kInt32);
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32);

    const int num_seqs = query.size(0);
    const int num_heads = query.size(1);
    const int head_size = query.size(2);

    const int num_kv_heads = key_cache.size(1);

    dim3 grid(num_seqs, num_heads);
    int NUM_WARPS = 4;
    dim3 block(NUM_WARPS * WARP_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        query.scalar_type(),
        "paged_attention_forward",
        [&]{
            paged_attention_kernel<scalar_t><<<grid, block, block_size * max_num_blocks_per_seq * sizeof(scalar_t), stream>>>(
                out.data_ptr<scalar_t>(),
                query.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                block_tables.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                scale,
                num_heads,
                head_size,
                num_kv_heads,
                block_size,
                max_num_blocks_per_seq
            );
        }
    );
    CHECK_KERNEL();
}