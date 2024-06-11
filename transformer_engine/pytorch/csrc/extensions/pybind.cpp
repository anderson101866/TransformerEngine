/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <pybind11/functional.h>

#include "common/userbuffers/comm_gemm_overlap.h"
#include "common/util/pybind_helper.h"

#include "../comm_gemm_overlap.h"
#include "../extensions.h"
#include "../common.h"


namespace te = transformer_engine;
namespace te_cgo = te::comm_gemm_overlap;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Load nvte = py::module_::import("transformer_engine_common") into TE/PyTorch. This makes
  // essential NVTE enums available through `import transformer_engine_torch` without requiring an
  // additional `import transformer_engine_common`.
  NVTE_ADD_COMMON_PYBIND11_BINDINGS(m)

  // Softmax functions
  m.def("scaled_softmax_forward", &scaled_softmax_forward, "Scaled Softmax FWD");
  m.def("scaled_softmax_backward", &scaled_softmax_backward, "Scaled Softmax BWD");
  m.def("scaled_masked_softmax_forward", &scaled_masked_softmax_forward,
                                                    "Scaled Masked Softmax FWD");
  m.def("scaled_masked_softmax_backward", &scaled_masked_softmax_backward,
                                                    "Scaled Masked Softmax BWD");
  m.def("scaled_upper_triang_masked_softmax_forward",
            &scaled_upper_triang_masked_softmax_forward,
            "Scaled Upper-Triangular Masked Softmax FWD");
  m.def("scaled_upper_triang_masked_softmax_backward",
            &scaled_upper_triang_masked_softmax_backward,
            "Scaled Upper-Triangular Masked Softmax BWD");
  m.def("scaled_aligned_causal_masked_softmax_forward",
            &scaled_aligned_causal_masked_softmax_forward,
            "Scaled Bottom-Right Corner Aligned Masked Softmax FWD");
  m.def("scaled_aligned_causal_masked_softmax_backward",
            &scaled_aligned_causal_masked_softmax_backward,
            "Scaled Bottom-Right Corner Aligned Masked Softmax BWD");

  // Other granular functions
  m.def("layernorm_fwd_fp8", &layernorm_fwd_fp8, "LN FWD FP8");
  m.def("layernorm_fwd_fp8_noalloc", &layernorm_fwd_fp8_noalloc, "LN FWD FP8");
  m.def("layernorm_bwd", &layernorm_bwd, "LN BWD");
  m.def("layernorm_fwd", &layernorm_fwd, "LN FWD");
  m.def("layernorm_fwd_noalloc", &layernorm_fwd_noalloc, "LN FWD");
  m.def("rmsnorm_fwd_fp8", &rmsnorm_fwd_fp8, "RMSNorm FWD FP8");
  m.def("rmsnorm_fwd_fp8_noalloc", &rmsnorm_fwd_fp8_noalloc, "RMSNorm FWD FP8");
  m.def("rmsnorm_bwd", &rmsnorm_bwd, "RMSNorm BWD");
  m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm FWD");
  m.def("rmsnorm_fwd_noalloc", &rmsnorm_fwd_noalloc, "RMSNorm FWD");
  m.def("fused_cast_transpose", &fused_cast_transpose, "Fused Cast + Transpose");
  m.def("fused_cast_transpose_noop", &fused_cast_transpose_noop,
                                              "Fused Cast + Transpose with noop option");
  m.def("fused_cast_transpose_bgrad", &fused_cast_transpose_bgrad,
                                              "Fused Cast + Transpose + BGRAD");
  m.def("fused_fp8_transpose_bgrad", &fused_fp8_transpose_bgrad,
                                              "Fused FP8 Transpose + BGRAD");
  m.def("fused_cast_transpose_bgrad_dgelu", &fused_cast_transpose_bgrad_dgelu,
                                              "Fused Cast + Transpose + BGRAD + DGELU");
  m.def("fused_multi_cast_transpose", &fused_multi_cast_transpose,
                                              "Fused Multi-tensor Cast + Transpose");
  m.def("cast_to_fp8", &cast_to_fp8, "Cast to FP8");
  m.def("cast_to_fp8_noalloc", &cast_to_fp8_noalloc, "Cast to FP8");
  m.def("cast_from_fp8", &cast_from_fp8, "Cast from FP8");
  m.def("te_gemm", &te_gemm, "CublasLt GEMM");
  m.def("fused_attn_fwd_qkvpacked", &fused_attn_fwd_qkvpacked,
                  "Fused Attention FP8/BF16/FP16 FWD with packed QKV");
  m.def("fused_attn_bwd_qkvpacked", &fused_attn_bwd_qkvpacked,
                  "Fused Attention FP8/BF16/FP16 BWD with packed QKV");
  m.def("fused_attn_fwd_kvpacked", &fused_attn_fwd_kvpacked,
                  "Fused Attention FP8/BF16/FP16 FWD with packed KV");
  m.def("fused_attn_bwd_kvpacked", &fused_attn_bwd_kvpacked,
                  "Fused Attention FP8/BF16/FP16 BWD with packed KV");
  m.def("fused_attn_fwd", &fused_attn_fwd,
                  "Fused Attention FP8/BF16/FP16 FWD with separate Q, K and V");
  m.def("fused_attn_bwd", &fused_attn_bwd,
                  "Fused Attention FP8/BF16/FP16 BWD with separate Q, K and V");
  m.def("fp8_transpose", &fp8_transpose, "Transpose with FP8 I/O");
  m.def("fp8_transpose_noalloc", &fp8_transpose_noalloc, "Transpose with FP8 I/O");
  m.def("fp8_transpose_noalloc_noop", &fp8_transpose_noalloc_noop,
                            "Transpose with FP8 I/O with noop option.");
  m.def("gelu", &gelu, "GeLU with FP8 output");
  m.def("relu", &relu, "ReLU with FP8 output");
  m.def("geglu", &geglu, "GeGLU with FP8 output");
  m.def("reglu", &reglu, "ReGLU with FP8 output");
  m.def("swiglu", &swiglu, "SwiGLU with FP8 output");
  m.def("qgelu", &qgelu, "QuickGELU with FP8 output");
  m.def("srelu", &srelu, "Squared ReLU with FP8 output");
  m.def("dgelu", &dgelu, "Backward of GeLU");
  m.def("drelu", &drelu, "Backward of ReLU");
  m.def("dgeglu", &dgeglu, "Backward of GeGLU");
  m.def("dreglu", &dreglu, "Backward of ReGLU");
  m.def("dswiglu", &dswiglu, "Backward of SwiGLU");
  m.def("dqgelu", &dqgelu, "Backward of QuickGELU");
  m.def("dsrelu", &dsrelu, "Backward of Squared ReLU");
  m.def("fa_prepare_fwd", &fa_prepare_fwd, "Prepare QKV for Flash Attention");
  m.def("fa_prepare_bwd", &fa_prepare_bwd, "Backward of QKV preparation for Flash Attention");
  m.def("get_fused_attn_backend", &get_fused_attn_backend, "Get Fused Attention backend");
  m.def("fused_amax_and_scale_update_after_reduction",
        &fused_amax_and_scale_update_after_reduction,
        "Update amax history and FP8 scale/scale_inv after reduction");

  // fused apply rope
  m.def("fused_rope_forward", &fused_rope_forward, "Fused Apply RoPE FWD");
  m.def("fused_rope_backward", &fused_rope_backward, "Fused Apply RoPE BWD");
  m.def("fused_rope_thd_forward", &fused_rope_thd_forward, "Fused Apply RoPE FWD for thd format");
  m.def("fused_rope_thd_backward", &fused_rope_thd_backward, "Fused Apply RoPE BWD for thd format");

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");
  m.def("get_cudnn_version", &get_cudnn_version, "Get cuDNN version");

  // Support THD format for Context Parallel
  m.def("thd_read_half_tensor", &thd_read_half_tensor,
        "Read the first half(half_idx=0) or the second half(half_idx=1) of each sequence in a THD "
        "tensor");
  m.def("thd_second_half_lse_correction", &thd_second_half_lse_correction,
        "Correct the second half of the softmax_lse");
  m.def("thd_read_second_half_lse", &thd_read_second_half_lse,
        "Read the second half of the softmax_lse");
  m.def("thd_out_correction", &thd_out_correction,
        "Correct the THD format output of context parallelism in forward pass");
  m.def("thd_grad_correction", &thd_grad_correction,
        "Correct the THD format gradients of context parallelism in backward pass");
  m.def("thd_get_partitioned_indices", &thd_get_partitioned_indices,
        "Generate partitioned indices for inputs in THD format");

  // multi-tensor functions
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_unscale_l2norm", &multi_tensor_unscale_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors after unscaling (unscaling is only "
        "performed for L2 norm computation, and tensors are not updated)");
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_adam_capturable", &multi_tensor_adam_capturable_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support and LR scheduling");
  m.def("multi_tensor_adam_capturable_master", &multi_tensor_adam_capturable_master_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support, LR scheduling and FP32 master weights");
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors");

  // Data structures
  py::class_<te::FP8TensorMeta>(m, "FP8TensorMeta", py::module_local())
    .def(py::init<>())
    .def_readwrite("scale", &te::FP8TensorMeta::scale)
    .def_readwrite("scale_inv", &te::FP8TensorMeta::scale_inv)
    .def_readwrite("amax_history", &te::FP8TensorMeta::amax_history);

  py::enum_<te::FP8FwdTensors>(m, "FP8FwdTensors", py::module_local())
    .value("GEMM1_INPUT", te::FP8FwdTensors::GEMM1_INPUT)
    .value("GEMM1_WEIGHT", te::FP8FwdTensors::GEMM1_WEIGHT)
    .value("GEMM1_OUTPUT", te::FP8FwdTensors::GEMM1_OUTPUT)
    .value("GEMM2_INPUT", te::FP8FwdTensors::GEMM2_INPUT)
    .value("GEMM2_WEIGHT", te::FP8FwdTensors::GEMM2_WEIGHT)
    .value("GEMM2_OUTPUT", te::FP8FwdTensors::GEMM2_OUTPUT)
    .value("GEMM3_INPUT", te::FP8FwdTensors::GEMM3_INPUT)
    .value("GEMM3_WEIGHT", te::FP8FwdTensors::GEMM3_WEIGHT)
    .value("GEMM3_OUTPUT", te::FP8FwdTensors::GEMM3_OUTPUT);

  py::enum_<te::FP8BwdTensors>(m, "FP8BwdTensors", py::module_local())
    .value("GRAD_OUTPUT1", te::FP8BwdTensors::GRAD_OUTPUT1)
    .value("GRAD_INPUT1", te::FP8BwdTensors::GRAD_INPUT1)
    .value("GRAD_OUTPUT2", te::FP8BwdTensors::GRAD_OUTPUT2)
    .value("GRAD_INPUT2", te::FP8BwdTensors::GRAD_INPUT2)
    .value("GRAD_OUTPUT3", te::FP8BwdTensors::GRAD_OUTPUT3)
    .value("GRAD_INPUT3", te::FP8BwdTensors::GRAD_INPUT3);

  // Comm+GEMM Overlap
  m.def("set_bootstrap_callbacks", &te_cgo::set_bootstrap_callbacks);

  py::class_<te_cgo::UbufCommOverlap>(m, "UbufCommOverlap", py::module_local())
    .def(py::init<torch::Tensor &, int, int, int, int, int, int, int, int, bool, bool>())
    .def("bulk_overlap", &te_cgo::UbufCommOverlap::bulk_overlap)
    .def("split_overlap_rs", &te_cgo::UbufCommOverlap::split_overlap_rs)
    .def("atomic_gemm_overlap_rs", &te_cgo::UbufCommOverlap::atomic_gemm_overlap_rs)
    .def("copy_input_to_ubuf", &te_cgo::UbufCommOverlap::copy_input_to_ubuf)
    .def("get_ubuf_output", &te_cgo::UbufCommOverlap::get_ubuf_output)
    .def("set_ubuf_scale_inv", &te_cgo::UbufCommOverlap::set_ubuf_scale_inv)
    .def("is_fp8_ubuf", &te_cgo::UbufCommOverlap::is_fp8_ubuf)
    .def("is_atomic_gemm", &te_cgo::UbufCommOverlap::is_atomic_gemm)
    .def("is_p2p_overlap", &te_cgo::UbufCommOverlap::is_p2p_overlap);

  py::class_<te_cgo::UbufP2PCommOverlap>(m, "UbufP2PCommOverlap", py::module_local())
    .def(py::init<torch::Tensor &, int, int, int, int, int, bool, bool, bool, bool>())
    .def("split_overlap_ag_p2p", &te_cgo::UbufP2PCommOverlap::split_overlap_ag)
    .def("split_overlap_rs_p2p", &te_cgo::UbufP2PCommOverlap::split_overlap_rs)
    .def("atomic_gemm_overlap_ag_p2p", &te_cgo::UbufP2PCommOverlap::atomic_gemm_overlap_ag)
    .def("atomic_gemm_overlap_rs_p2p", &te_cgo::UbufP2PCommOverlap::atomic_gemm_overlap_rs)
    .def("copy_input_to_ubuf", &te_cgo::UbufP2PCommOverlap::copy_input_to_ubuf)
    .def("get_ubuf_output", &te_cgo::UbufP2PCommOverlap::get_ubuf_output)
    .def("set_ubuf_scale_inv", &te_cgo::UbufP2PCommOverlap::set_ubuf_scale_inv)
    .def("is_fp8_ubuf", &te_cgo::UbufP2PCommOverlap::is_fp8_ubuf)
    .def("is_atomic_gemm", &te_cgo::UbufP2PCommOverlap::is_atomic_gemm)
    .def("is_p2p_overlap", &te_cgo::UbufP2PCommOverlap::is_p2p_overlap);
}
