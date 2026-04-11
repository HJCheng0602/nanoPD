#include"paged_kv_store.h"
#include"../utils/cuda_utils.h"
#include"../utils/type_utils.h"


// let's think the kernel, first we copy the signiture from the .h
// ok, we start to think 
// the k_cache shape is (max_blocks, num_kv_heads, block_size, head_dim)
// meanwhile the k_src dim is (num_kv_heads, seq_len, head_dim)

// our launch param is grid(seq_len, num_kv_heads)      block(head_dim)
// naturally one thread is resibonsible for a adsfadsfa
// block_size is 


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
)
{
    int tid     = threadIdx.x;
    int head_id = blockIdx.y;
    int seq_id  = blockIdx.x;

    const int vec_head_dim = head_dim / 4;

    float2 *       k_cache_f2 = reinterpret_cast<float2*>(k_cache);
    float2 *       v_cache_f2 = reinterpret_cast<float2*>(v_cache);
    const float2 * k_src_f2   = reinterpret_cast<const float2*>(k_src);
    const float2 * v_src_f2   = reinterpret_cast<const float2*>(v_src);

    // src: (num_kv_heads, num_tokens, head_dim)
    int src_idx = head_id * num_tokens * vec_head_dim
                + seq_id * vec_head_dim
                + tid;

    int position     = positions[seq_id];
    int block_idx    = position / block_size;
    int block_offset = position % block_size;

    int physical_block = block_tables[seq_id * max_blocks_per_seq + block_idx];

    int dst_idx = physical_block * num_kv_heads * block_size * vec_head_dim
                + head_id        * block_size    * vec_head_dim
                + block_offset   * vec_head_dim
                + tid;

    k_cache_f2[dst_idx] = k_src_f2[src_idx];
    v_cache_f2[dst_idx] = v_src_f2[src_idx];
}