/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#if __cplusplus >= 201703L
  #include <optional>
#elif __cplusplus >= 201402L
  #include <experimental/optional>
  #ifndef PYBIND11_HAS_EXP_OPTIONAL
    #define PYBIND11_HAS_EXP_OPTIONAL 1 //paddle's dependency define <optional> in c++17, which confuse pybind11 from correctly defining PYBIND11_HAS_EXP_OPTIONAL. Here forcely define it
  #endif
  #include <pybind11/stl.h>
  #define EXP_OPTIONAL_OF_TENSOR
#else
  #error "__cplusplus is undefined!"
#endif

#include "common.h"
#include "common/util/pybind_helper.h"
#include "comm_gemm_overlap.h"
#include <memory>

namespace transformer_engine {
namespace paddle_ext {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

namespace te = transformer_engine;
namespace te_cgo = te::paddle_ext::comm_gemm_overlap;

PYBIND11_MODULE(transformer_engine_paddle, m) {
  // Load nvte = py::module_::import("transformer_engine_common") into TE/Paddle. This makes
  // essential NVTE enums available through `import transformer_engine_paddle` without requiring
  // an additional `import transformer_engine_common as tex`.
  NVTE_ADD_COMMON_PYBIND11_BINDINGS(m)
  
  // Comm+GEMM Overlap
  py::class_<te_cgo::PaddleDistributedCallbackHolder>(m, "PaddleDistributedCallbackHolder")
    .def(py::init<>());
  m.attr("_dist_callback_holder") = py::cast(
    std::make_unique<te_cgo::PaddleDistributedCallbackHolder>(), 
    py::return_value_policy::take_ownership //module m track its lifecycle
  );
  m.def("set_comm_overlap_callbacks", &te_cgo::set_comm_overlap_callbacks,
    py::arg("callback_holder").none(false), //to reject None
    py::arg("allgather"),
    py::arg("bcast"),
    py::arg("barrier")
  );
  
  py::class_<te_cgo::CommGemmOverlapP2P>(m, "CommGemmOverlapP2P", py::module_local())
      .def(py::init<const paddle::Tensor & /* sample */, int /* world_rank */, int /* world_size */, int /* local_rank */, 
                    int /* local_size */, int /* node_id */, int /* num_nodes */, int /* num_max_streams */, int /* tp_size */, 
                    int /* cga_size */, int /* num_comm_sms */, bool /* set_sm_margin */, bool /* use_ce */, 
                    bool /* atomic_gemm */, bool /* aggregate */, bool /* is_reduce_scatter */>())
      .def("split_overlap_ag_p2p", &te_cgo::CommGemmOverlapP2P::split_overlap_ag,
           py::call_guard<py::gil_scoped_release>())
      .def("split_overlap_rs_p2p", &te_cgo::CommGemmOverlapP2P::split_overlap_rs,
           py::call_guard<py::gil_scoped_release>())
      .def("copy_input_to_ubuf", &te_cgo::CommGemmOverlapP2P::copy_input_to_ubuf,
           py::call_guard<py::gil_scoped_release>())
      .def("get_ubuf_output", &te_cgo::CommGemmOverlapP2P::get_ubuf_output,
           py::call_guard<py::gil_scoped_release>())
      .def("set_ubuf_scale_inv", &te_cgo::CommGemmOverlapP2P::set_ubuf_scale_inv,
           py::call_guard<py::gil_scoped_release>())
      .def("is_fp8_ubuf", &te_cgo::CommGemmOverlapP2P::is_fp8_ubuf,
           py::call_guard<py::gil_scoped_release>())
      .def("is_atomic_gemm", &te_cgo::CommGemmOverlapP2P::is_atomic_gemm,
           py::call_guard<py::gil_scoped_release>())
      .def("is_p2p_overlap", &te_cgo::CommGemmOverlapP2P::is_p2p_overlap,
           py::call_guard<py::gil_scoped_release>());

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");
  m.def("get_fused_attn_backend", &get_fused_attn_backend, "Get Fused Attention backend");
  m.def("get_nvte_qkv_layout", &get_nvte_qkv_layout, "Get qkv layout enum by the string");
}
}  // namespace paddle_ext
}  // namespace transformer_engine
