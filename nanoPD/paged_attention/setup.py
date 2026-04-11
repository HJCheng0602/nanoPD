import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="paged_kernels",
    ext_modules=[
        CUDAExtension
        (name="paged_kernels",
        sources=[
            "csrc/attention/paged_attention_ops.cu",
            "csrc/attention/paged_attention_bindings.cpp",
            "csrc/kvstore/paged_kv_store_bindings.cpp",
            "csrc/kvstore/paged_kv_store_ops.cu"
        ],
        include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")],
        extra_compile_args={
            "cxx":["-O3"],
            "nvcc":[
                "-O3",
                "-arch=native",
                "--use_fast_math",
                "-lineinfo",
                ]
        })
    ],
    cmdclass={"build_ext":BuildExtension}
)