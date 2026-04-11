#include "paged_attention_optimized.h"
#include "../utils/cuda_utils.h"
#include "../utils/type_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include "paged_attention_optimized.cu"


#define DISPATCH_HEAD_SIZE(head_size, HEAD_SIZE, ...)   \
    if (head_size == 128)       { constexpr int HEAD_SIZE = 128; __VA_ARGS__; } \
    else if (head_size == 256)  { constexpr int HEAD_SIZE = 256; __VA_ARGS__; } \
    else { TORCH_CHECK(false, "unsupported head_size: ", head_size); }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)  \
    if (block_size == 16)       { constexpr int BLOCK_SIZE = 16; __VA_ARGS__; } \
    else if (block_size == 32)  { constexpr int BLOCK_SIZE = 32; __VA_ARGS__; } \
    else { TORCH_CHECK(false, "unsupported block_size: ", block_size); }


    
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
    TORCH_CHECK(query.is_cuda());
    TORCH_CHECK(query.dim() == 3);
    TORCH_CHECK(key_cache.dim() == 5);
    TORCH_CHECK(value_cache.dim() == 4);
    TORCH_CHECK(block_tables.dtype() == torch::kInt32);
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32);

    const int num_seqs     = query.size(0);
    const int num_heads    = query.size(1);
    const int head_size    = query.size(2);
    const int num_kv_heads = key_cache.size(1);

    int max_seq_len = seq_lens.max().item<int>();

    constexpr int PARTITION_SIZE = 512;
    constexpr int NUM_THREADS    = 128;
    int num_partitions = (max_seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

    auto partial_out = torch::empty(
        {num_seqs, num_heads, num_partitions, head_size},
        query.options());
    auto exp_sums = torch::empty(
        {num_seqs, num_heads, num_partitions},
        torch::dtype(torch::kFloat32).device(query.device()));
    auto max_logits = torch::empty(
        {num_seqs, num_heads, num_partitions},
        torch::dtype(torch::kFloat32).device(query.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid1(num_seqs, num_heads, num_partitions);
    dim3 block1(NUM_THREADS);
    dim3 grid2(num_seqs, num_heads);
    dim3 block2(NUM_THREADS);

    const auto dtype = query.scalar_type();
    TORCH_CHECK(
        dtype == at::ScalarType::Half ||
        dtype == at::ScalarType::BFloat16 ||
        dtype == at::ScalarType::Float,
        "paged_attention only supports float16, bfloat16, float32, got: ",
        dtype);

#define LAUNCH(scalar_t)                                                          \
    paged_attention_kernel                                                        \
        <scalar_t, HEAD_SIZE, 32, NUM_THREADS, PARTITION_SIZE, BLOCK_SIZE> \
        <<<grid1, block1, 0, stream>>>(                                           \
            partial_out.data_ptr<scalar_t>(),                                     \
            exp_sums.data_ptr<float>(),                                           \
            max_logits.data_ptr<float>(),                                         \
            query.data_ptr<scalar_t>(),                                           \
            key_cache.data_ptr<scalar_t>(),                                       \
            value_cache.data_ptr<scalar_t>(),                                     \
            block_tables.data_ptr<int>(),                                         \
            seq_lens.data_ptr<int>(),                                             \
            scale, num_heads, head_size,                                          \
            num_kv_heads, block_size,                                             \
            max_num_blocks_per_seq, num_partitions);                              \
    CHECK_KERNEL();                                                               \
    paged_attention_reduce_kernel                                                 \
        <scalar_t, HEAD_SIZE, NUM_THREADS>                                        \
        <<<grid2, block2, 0, stream>>>(                                           \
            out.data_ptr<scalar_t>(),                                             \
            partial_out.data_ptr<scalar_t>(),                                     \
            exp_sums.data_ptr<float>(),                                           \
            max_logits.data_ptr<float>(),                                         \
            num_heads, num_partitions, head_size);                                \
    CHECK_KERNEL();

    DISPATCH_HEAD_SIZE(head_size, HEAD_SIZE,
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            if (dtype == at::ScalarType::Half)
               { LAUNCH(c10::Half)}
            else if (dtype == at::ScalarType::BFloat16)
                {LAUNCH(c10::BFloat16)}
            else
                {LAUNCH(float)}
        })
    );

#undef LAUNCH
}