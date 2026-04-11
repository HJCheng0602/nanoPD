#include "paged_kv_store.h"
#include "../utils/cuda_utils.h"
#include "../utils/type_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include "paged_kv_store_kernel.cu"

void paged_kv_store(
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    const torch::Tensor& k_src, const torch::Tensor& v_src,
    const torch::Tensor& block_tables, const torch::Tensor& positions
)
{
    TORCH_CHECK(k_src.is_cuda(), "v src is must be on CUDA");
    TORCH_CHECK(v_src.is_cuda(), "v src is must be on CUDA");

    TORCH_CHECK(k_src.dim() == 3);

    const int num_kv_heads = k_src.size(0);
    const int seq_len = k_src.size(1);
    const int head_dim = k_src.size(2);

    const int block_size = k_cache.size(2);
    const int max_blocks_per_seq = block_tables.size(1);
    
    dim3 grid(seq_len, num_kv_heads);
    dim3 block(head_dim / 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k_cache.scalar_type(),
        "paged_kv_store",
        [&]{
            paged_kv_store_kernel<scalar_t><<<grid, block, 0, stream>>>(
                k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
                k_src.data_ptr<scalar_t>(), v_src.data_ptr<scalar_t>(),
                block_tables.data_ptr<int>(),
                positions.data_ptr<int>(),
                seq_len,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq
            );
        }
    );
    CHECK_KERNEL();
}