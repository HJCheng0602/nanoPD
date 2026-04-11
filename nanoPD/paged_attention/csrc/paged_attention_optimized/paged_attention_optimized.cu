#include"paged_attention_optimized.h"
#include"../utils/cuda_utils.h"
#include"../utils/type_utils.h"
#include"cache_utils.h"
#include"index.h"



template<int HEAD_SIZE>
__device__ __forceinline__ float warp_dot_scale(
    const float * q_reg,
    const float * k_reg,
    float scale,
    int lane_id
)
{
    float partial = 0.0f;
    constexpr int elems_per_lane = HEAD_SIZE / WARP_SIZE;

    for(int i = 0; i < elems_per_lane; i ++)
    {
        partial += q_reg[i] * k_reg[i];
    }

    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    return __shfl_sync(0xffffffff, partial * scale, 0);
}

template<typename scalar_t, int HEAD_SIZE, int WARP_SIZE>
__device__ __forceinline__ void load_k_for_token(
    float *k_reg, const scalar_t * k_cache,
    int physical_block_id, int kv_head_id, int token_in_block,
    int num_kv_heads, int head_size, int block_size, int lane_id
)
{
    constexpr int x = VecLoadTraits<scalar_t>::x;
    constexpr int elems_per_lane = HEAD_SIZE / WARP_SIZE;
    constexpr int num_groups = HEAD_SIZE / x;

    // each lane resibonsible for elems per len elements
    // lane_id -> lan_id * elem_per_lane
    int elem_start = lane_id * elems_per_lane;
    for(int i = 0; i < elems_per_lane; i ++)
    {
        int elem_idx = elem_start + i;
        int group_id = elem_idx / x;
        int in_group_offset = elem_idx % x;
        

        const scalar_t * ptr = KCacheLayout<scalar_t>::get_k_ptr(
            k_cache, physical_block_id, kv_head_id, token_in_block, group_id, num_kv_heads, head_size, block_size
        );

        k_reg[i] = to_float(ptr[in_group_offset]);
    }
}



// the grid is (seq_len, num_head)
// block is (NUM_THREAD)

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
)
{
    float q_reg[HEAD_SIZE / WARP_SIZE];
    float v_acc[HEAD_SIZE / NUM_THREADS];
    for (int i = 0; i < HEAD_SIZE / NUM_THREADS; i++) v_acc[i] = 0.0f;
    float l_max = -FLT_MAX;
    float m_sum = 0.0f;
    int seq_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int head_id = blockIdx.y;
    int kv_head_id = head_id / (num_heads / num_kv_heads);
    int partition_id = blockIdx.z;
    int token_start = partition_id * PARTITION_SIZE;
    int seq_len = seq_lens[seq_id];
    int token_end = min(token_start + PARTITION_SIZE, seq_len);
    


    {
        constexpr int elems_per_lane = HEAD_SIZE / WARP_SIZE;
        const scalar_t* q_ptr = q + seq_id * num_heads * head_size + head_id * head_size;
        for (int i = 0; i < elems_per_lane; i++)
            q_reg[i] = to_float(q_ptr[lane_id * elems_per_lane + i]);
    }
    
    
    
    
    int num_blocks = (seq_len + block_size - 1)  / block_size;
    
    int block_start = token_start / block_size;
    int block_end = (token_end + block_size - 1) / block_size;

    for(int block_id = block_start; block_id < block_end; block_id ++)
    {
        int physical_block_id = block_tables[seq_id * max_num_blocks_per_seq + block_id];

        for (int token_in_block = 0; token_in_block < BLOCK_SIZE; token_in_block ++)
        {
            int global_token_id = block_id * block_size + token_in_block;
            if(global_token_id >= token_end || global_token_id < token_start) continue;

            float k_reg[HEAD_SIZE / WARP_SIZE];
            load_k_for_token<scalar_t, HEAD_SIZE, WARP_SIZE>(
                k_reg, k_cache, physical_block_id, kv_head_id,
                token_in_block,num_kv_heads, head_size, block_size, lane_id
            );
            float score = warp_dot_scale<HEAD_SIZE>(q_reg, k_reg, scale, lane_id);

            float l_max_new = fmaxf(l_max, score);
            float alpha = expf(l_max - l_max_new);
            float p = expf(score - l_max_new);

            for(int i = 0; i < HEAD_SIZE / NUM_THREADS; i ++)
            {
                v_acc[i] *= alpha;
            }

            m_sum = m_sum * alpha + p;
            l_max = l_max_new;

            for(int i = 0; i < HEAD_SIZE / NUM_THREADS; i ++)
            {
                int h_idx = tid * (HEAD_SIZE / NUM_THREADS) + i;
                const scalar_t* v_ptr = VCacheLayout<scalar_t>::get_ptr(
                    v_cache, physical_block_id, kv_head_id, h_idx, token_in_block, num_kv_heads, head_size, block_size
                );
                v_acc[i] += p * to_float(*v_ptr);
            }
            
        }
    }

    for(int i = 0; i < HEAD_SIZE / NUM_THREADS; i ++)
    {
        int h_idx = tid * (HEAD_SIZE / NUM_THREADS) + i;
        partial_out[seq_id * num_partition * num_heads * head_size + head_id * head_size * num_partition + partition_id * head_size + h_idx] = from_float<scalar_t>(v_acc[i]);
    }

    if(tid == 0)
    {
        int idx = seq_id * num_heads * num_partition 
            + head_id * num_partition
            + partition_id;
        exp_sums[idx] = m_sum;
        max_logits[idx] = l_max;
    }
}

template<typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__global__ void paged_attention_reduce_kernel(
    scalar_t* out,                    // [num_seqs, num_heads, head_size]
    const scalar_t* partial_out,      // [num_seqs, num_heads, num_partitions, head_size]
    const float* exp_sums,            // [num_seqs, num_heads, num_partitions]
    const float* max_logits,          // [num_seqs, num_heads, num_partitions]
    int num_heads, int num_partitions, int head_size)
{
    int seq_id  = blockIdx.x;
    int head_id = blockIdx.y;
    int tid     = threadIdx.x;

    float global_max = -FLT_MAX;
    for(int p = 0; p < num_partitions; p++) {
        int idx = seq_id * num_heads * num_partitions + head_id * num_partitions + p;
        global_max = fmaxf(global_max, max_logits[idx]);
    }

    float global_exp_sum = 0.0f;
    for(int p = 0; p < num_partitions; p++) {
        int idx = seq_id * num_heads * num_partitions + head_id * num_partitions + p;
        global_exp_sum += exp_sums[idx] * expf(max_logits[idx] - global_max);
    }

    float acc[HEAD_SIZE / NUM_THREADS] = {0.0f};

    for(int p = 0; p < num_partitions; p++) {
        int meta_idx = seq_id * num_heads * num_partitions + head_id * num_partitions + p;
        float rescale = expf(max_logits[meta_idx] - global_max)
                      / global_exp_sum;

        for (int i = 0; i < HEAD_SIZE / NUM_THREADS; i++) {
            int h_idx = tid * (HEAD_SIZE / NUM_THREADS) + i;
            int out_idx = seq_id * num_heads * num_partitions * head_size
                        + head_id * num_partitions * head_size
                        + p * head_size
                        + h_idx;
            acc[i] += rescale * to_float(partial_out[out_idx]);
        }
    }

    for (int i = 0; i < HEAD_SIZE / NUM_THREADS; i++) {
        int h_idx = tid * (HEAD_SIZE / NUM_THREADS) + i;
        out[seq_id * num_heads * head_size + head_id * head_size + h_idx]
            = from_float<scalar_t>(acc[i]);
    }
};
