#pragma once
#include <ATen/cuda/CUDAContext.h>
#include "../utils/type_utils.h"
template <typename scalar_t>

struct VecLoadTraits;

template<>
struct VecLoadTraits<c10::Half> {
    static constexpr int x = 8;
    using vec_t = uint4;
    using native_t = __half;  
};
template<>
struct VecLoadTraits<c10::BFloat16> {
    static constexpr int x = 8;
    using vec_t = uint4;
    using native_t = __nv_bfloat16;
};

template<>
struct VecLoadTraits<float>
{
    static constexpr int x = 4;
    using vec_t = uint4;
};

template<typename scalar_t>
__device__ __forceinline__ void load_vec(float* dst, const scalar_t* src)
{
    using Traits = VecLoadTraits<scalar_t>;
    using vec_t  = typename Traits::vec_t;

    vec_t packed = __ldg(reinterpret_cast<const vec_t*>(src));

    union { vec_t raw; scalar_t vals[Traits::x]; } u;
    u.raw = packed;
    for (int i = 0; i < Traits::x; i++)
        dst[i] = to_float(u.vals[i]);
}

template<typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__device__ __forceinline__ void load_head(
    float * dst,
    const scalar_t* src,
    int tid
)
{
    using Traits = VecLoadTraits<scalar_t>;
    constexpr int x = Traits::x;
    constexpr int elems_per_t = HEAD_SIZE / NUM_THREADS;
    constexpr int vecs_per_t = elems_per_t / x;

    const scalar_t * base = src + tid * elems_per_t;

    for(int v = 0; v < vecs_per_t; v++)
    {
        load_vec<scalar_t>(dst + v * x, base + v * x);
    }
}