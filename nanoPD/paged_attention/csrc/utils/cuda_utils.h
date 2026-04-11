#pragma once
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

constexpr int WARP_SIZE = 32;

#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t err = (call);                   \
        if(err != cudaSuccess){                     \
        fprintf(stderr, "CUDA error at %s:%d - %s\n"\
                __FILE__, __LINE__, cudaGetErrorString(err)); \
                exit(EXIT_FAILURE);                             \
        }                                                        \
    } while(0)                                                 \

inline void check_kernel_launch(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error at %s:%d - %s\n",
            file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_KERNEL() check_kernel_launch(__FILE__, __LINE__)