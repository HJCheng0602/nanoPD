#pragma once

template <typename T>
__device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }
