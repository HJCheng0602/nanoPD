#include<torch/extension.h>
#include"kvstore/paged_kv_store.h"

void register_paged_kv_store(py::module& m)
{
    m.def(
        "paged_kv_store",
        &paged_kv_store,
        "Paged kv store(CUDA)"
    );
}
