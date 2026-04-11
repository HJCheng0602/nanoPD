#include<torch/extension.h>
#include"paged_attention_optimized/paged_attention_optimized.h"

void register_paged_kv_store(py::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "paged_attention_forward",
        &paged_attention_forward,
        "Paged attetnion forward pass(CUDA)"
    );
    register_paged_kv_store(m);
}

