#pragma once


// we think k is (.., headsize / x, blocksize, x) instead of (..., blocksize, headsize/x, x)
// the reason is that when we consider different threads in a warp:
// thread 0 read token t, group j， thread 1 read token t + 1, j
// if the second, it will read t * (headsize / x * x) + j * x = t * headsize + j * x
// if first, it will read j * (blocksize * x) + t * x 
// so the first can allow next thread read next x

// so we rule that the k layout is like (.., headsize / x, blocksize, x)


template <typename scalar_t>
struct KCacheLayout
{
    static constexpr int x = 16 / sizeof(scalar_t);

    static __device__ __forceinline__ const scalar_t* get_k_ptr(
        const scalar_t * k_cache, 
        int block_idx, int kv_head_id, int token_in_block, int group_id, int num_kv_heads, int head_size, int block_size
    )
    {
        int head_groups = head_size / x;
        return k_cache 
                + block_idx * (num_kv_heads * head_groups * block_size * x)
                + kv_head_id * (head_groups * block_size * x)
                + group_id * (block_size * x) 
                + token_in_block * x;
    }
};


// for v
// let's think the whole process when we use v
// out[h] += softmax_score[tokenid] * v[tokenid][h]
// here we think that each thread's job

// when each thread tackle different tokens, and sum them up, it is better

// so here for the initial layout (num_blocks, num_kv_heads, block_size, head_size)
// token0: v00 v01 v02 v03 v04 v05 v06 v07
// token1: v10 v11 v12 v13 v14 v15 v16 v17
// token2: v20 v21 v22 v23 v24 v25 v26 v27
// token3: v30 v31 v32 v33 v34 v35 v36 v37

// thread0 tackle v00 v01 v02,  a good situation
// but, let's see a warp, t0 for v00, t1 for v10... bad
// it is not suitable for the prefetcher and the cache line,
// so the good way is to (num_blocks, num_kv_heads, head_size, block_size)
// v00 v10 v20 v30
// v01 v11 v21 v31
// ...
// ...
// ...
// suitable for cache line use
// here t0 for v00, t1 for v10, good use
// for each token, they will load a same line, good for cacheline

template<typename scalar_t>
struct VCacheLayout
{
    static __device__ __forceinline__ const scalar_t* get_ptr(
        const scalar_t* v_cache,
        int block_idx, int kv_head_id, int h_idx, int token_in_block,
        int num_kv_heads, int head_size, int block_size
    )
    {
        return v_cache 
                + block_idx * (num_kv_heads * head_size * block_size)
                + kv_head_id * (head_size * block_size)
                + h_idx * block_size
                + token_in_block;
    }
};

