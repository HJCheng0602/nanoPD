#include"paged_attention.h"
#include"../utils/cuda_utils.h"
#include"../utils/type_utils.h"


// warp_in_dotmul must be defined before paged_attention_kernel
template <typename scalar_t>
__device__ void warp_in_dotmul(
                scalar_t * out,
                const scalar_t * a_line,
                const scalar_t * b_line,
                int size,
                int lane_id,
                float scale)
{
    int elements_per_thread = size / WARP_SIZE;
    float partial_sum = 0.0f;
    for(int i = 0; i < elements_per_thread; i ++)
    {
        partial_sum += to_float<scalar_t>(a_line[lane_id + i * WARP_SIZE]) * to_float<scalar_t>(b_line[lane_id + i * WARP_SIZE]);
    }

    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    if(lane_id == 0)
    {
        *out = from_float<scalar_t>(partial_sum * scale);
    }
}


//pay attention that the key_cache is the size like [num_physical_blocks, num_kv_heads, blocksize, headsize]
//                       value_cache is the size like [num_physical_blocks, num_kv_heads, blocksize, headsize]
// query is the shape like [num_seq, num_heads, head_size]

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
)
{
    // Use char shared memory to avoid redeclaration conflict across template instantiations
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    float v_acc[8] = {0.0f};
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int kv_head_id = head_id * num_kv_heads /  num_heads;
    int tid = threadIdx.x;
    int token_id = tid/ WARP_SIZE;
    int lane_id = tid % WARP_SIZE;


    // here we stop to think about the kernel's job
    // our ideal out is [num_seq, num_heads, head_size]
    //
    // every block will responsible for [1, 1, head_size]
    // how to compute it?
    const scalar_t * Q_data_ptr = q + seq_id * num_heads * head_size + head_id * head_size;
    int seq_len = *(seq_lens + seq_id);
    int block_nums = (seq_len + block_size - 1) / block_size;
    const int * block_idxes = block_tables + seq_id * max_num_blocks_per_seq;
    float l_max = -3e38;
    float m_sum = 0;

    for(int block_id = 0; block_id < block_nums; block_id ++)
    {
        int physical_block_id = *(block_idxes + block_id);
        const scalar_t * K_cache_location = k_cache + physical_block_id * num_kv_heads * block_size * head_size + \
                                    kv_head_id * block_size * head_size;
        // then for a token in a block, we choose a warp to tackle the data, a warp(32 thread) tackle head_size elements
        for(int roll_out = 0; roll_out < block_size; roll_out += (blockDim.x / WARP_SIZE))
        {
            if(block_id * block_size + roll_out + token_id < seq_len)
            {
                const scalar_t * K_cache_location_of_a_warp = K_cache_location + (roll_out + token_id) * head_size;
                // then a warp tackle head_size's dot mul
                warp_in_dotmul(sdata + block_id * block_size + roll_out + token_id , Q_data_ptr, K_cache_location_of_a_warp, head_size, lane_id, scale);
            }
            __syncthreads();
            // I want to unroll the add
            for(int temp = 0; temp < (blockDim.x / WARP_SIZE); temp ++)
            {
                if (block_id * block_size + roll_out + temp < seq_len) {
                    float nowdata = to_float<scalar_t>(sdata[block_id * block_size + roll_out + temp]);
                    float l_max_new = max(l_max, nowdata);
                    m_sum = m_sum * exp(l_max - l_max_new) + exp(nowdata - l_max_new);
                    l_max = l_max_new;
                }
            }
        }
    }
    for(int sdata_idx = 0; sdata_idx < seq_len; sdata_idx += blockDim.x)
    {
        int current_idx = sdata_idx + tid;
        if (current_idx < seq_len) {
            float val = to_float<scalar_t>(sdata[current_idx]);
            sdata[current_idx] = from_float<scalar_t>(exp(val - l_max) / m_sum);
        }
    }
    __syncthreads();
    int element_per_thread_v = (head_size + blockDim.x - 1) / blockDim.x;

    for(int i = 0; i < element_per_thread_v; i ++)
    {
        v_acc[i] = 0.0f;
    }

    for(int i = 0; i < seq_len; i ++)
    {
        float p = to_float<scalar_t>(sdata[i]);
        int logical_block_id = i / block_size;
        int offset_in_block = i % block_size;
        int physical_block_id = *(block_idxes + logical_block_id);

        const scalar_t * V_ptr = v_cache + physical_block_id * num_kv_heads * block_size * head_size + kv_head_id * block_size * head_size + offset_in_block * head_size;

        for(int step = 0; step < element_per_thread_v; step ++)
        {
            int h_idx = tid + step * blockDim.x;
            if(h_idx < head_size)
            {
                v_acc[step] += p * to_float<scalar_t>(V_ptr[h_idx]);
            }
        }
    }

    for(int step = 0; step < element_per_thread_v; step++)
    {
        int h_idx = tid + step * blockDim.x;
        if(h_idx < head_size)
        {
            int out_offset = seq_id * num_heads * head_size + head_id * head_size + h_idx;
            out[out_offset] = from_float<scalar_t>(v_acc[step]);
        }
    }
}
