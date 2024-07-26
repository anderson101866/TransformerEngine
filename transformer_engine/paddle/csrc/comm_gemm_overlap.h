/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

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

#include <functional>
#include <vector>
#include <tuple>
#include <pybind11/functional.h>

#include "paddle/extension.h"
#include <transformer_engine/comm_gemm_overlap.h>
 
namespace transformer_engine {
namespace paddle_ext {
namespace comm_gemm_overlap {

//using optional_tensor_ref = paddle::optional<paddle::Tensor>&;
//using optional_tensor_ref = paddle::Tensor*;
#ifndef EXP_OPTIONAL_OF_TENSOR
  using optional_tensor_ref = std::optional<paddle::Tensor>;
#else
  using optional_tensor_ref = std::experimental::optional<paddle::Tensor>;
#endif

/**
 * Static container for Python callbacks to paddle.distributed collectives, 
 * and holds the python Tensor object they're using.
*/
struct PaddleDistributedCallbackHolder {
public:
  PaddleDistributedCallbackHolder() = default;
  std::function<void(/*out*/paddle::Tensor &, const paddle::Tensor &, const std::string &)> allgather;
  std::function<void(/*out*/paddle::Tensor &, int64_t, const std::string &)> bcast;
  std::function<void(const std::string &)> barrier;
};

/**
 * Helper function for setting Python callbacks to torch.distributed collectives.
 */
void set_comm_overlap_callbacks(PaddleDistributedCallbackHolder *callback_holder, 
  const std::function<void(/*out*/paddle::Tensor &, const paddle::Tensor &, const std::string &)> &allgather,
  const std::function<void(/*out*/paddle::Tensor &, int64_t, const std::string &)> bcast,
  const std::function<void(const std::string &)> &barrier);

/** 
 * Userbuffer-based implementation of GEMM. Each overlapping algorithm is implemented in each method.
 * NOTE: Since paddle 2.6 doesn't support to bind C++ class as customized operator
 *  but only support C function via `PD_BUILD_OP`,
 *  We expose directly this class as python cpp-extension through pybind's py::class_ interface.
 */
class CommGemmOverlapP2P : 
  public transformer_engine::common::CommGemmOverlapP2P {  
public:
  CommGemmOverlapP2P(const paddle::Tensor &sample, int world_rank, int world_size, int local_rank, int local_size, 
                     int node_id, int num_nodes, int tp_size, int num_max_streams, int cga_size, int num_comm_sms, 
                     bool set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate, bool is_reduce_scatter);
  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(paddle::Tensor &scale_inv) {
    _ubuf_scale_inv_ptr = reinterpret_cast<float *>(scale_inv.data());
    _ubuf_scale_inv_initialized = true;
  }

  /**
   * Split AllGather + Pipelined GEMM using P2P communication
   * This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
   * outputs in each rank to be in the contiguous memory space after all ring exchange phases.
   */
  void split_overlap_ag(const paddle::Tensor &A, const optional_tensor_ref A_scale_inverse,
                        const paddle::Tensor &B, const optional_tensor_ref B_scale_inverse,
                        const optional_tensor_ref bias, paddle::Tensor &D,           // NOLINT
                        optional_tensor_ref D_scale,                                 // NOLINT
                        optional_tensor_ref D_amax,                                  // NOLINT
                        optional_tensor_ref pre_gelu_out, paddle::Tensor &workspace, // NOLINT
                        optional_tensor_ref B_copy,                                  // NOLINT
                        int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
                        int64_t D_type, int64_t bias_type, bool transa, bool transb, bool grad,
                        int64_t workspace_size, bool accumulate, bool use_split_accumulator);

  /**
   * Pipelined GEMM + Split Reduce+Scatter using P2P communication
   */
  void split_overlap_rs(const paddle::Tensor &A, const optional_tensor_ref A_scale_inverse,
                        const paddle::Tensor &B, const optional_tensor_ref B_scale_inverse,
                        const optional_tensor_ref bias, /*out*/paddle::Tensor &D,    // NOLINT
                        optional_tensor_ref D_scale,                                 // NOLINT
                        optional_tensor_ref D_amax,                                  // NOLINT
                        optional_tensor_ref pre_gelu_out, paddle::Tensor &workspace, // NOLINT
                        paddle::Tensor &rs_output,                                                 // NOLINT
                        int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
                        int64_t D_type, int64_t bias_type, bool transa, bool transb, bool grad,
                        int64_t workspace_size, bool accumulate, bool use_split_accumulator);

  /**
   * Helper function to copy input to _ubuf or _ubufs chunks.
   */
  void copy_input_to_ubuf(const paddle::Tensor &input, bool chunk);
  
  /**
   * Helper function to export _ubuf output by wrap its buffer with a `paddle::Tensor`
   * @param comm_type ALL_GATHER: treat the ubuf as already be all-gathered. REDUCE_SCATTER: already be reduce-scattered
   */
  paddle::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type);
  bool is_fp8_ubuf();
private:
  std::tuple<TensorWrapper/*A_*/, 
             TensorWrapper/*B_*/, 
             TensorWrapper/*D_*/, 
             TensorWrapper/*bias_*/,
             TensorWrapper/*pre_gelu_out_*/,
             TensorWrapper/*workspace_*/> _wrap_tensors(
    const paddle::Tensor &A, const optional_tensor_ref A_scale_inverse,
    const paddle::Tensor &B, const optional_tensor_ref B_scale_inverse,
    const optional_tensor_ref bias, /*out*/paddle::Tensor &D,    // NOLINT
    optional_tensor_ref D_scale,                                 // NOLINT
    optional_tensor_ref D_amax,                                  // NOLINT
    optional_tensor_ref pre_gelu_out, paddle::Tensor &workspace, // NOLINT
    int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
    int64_t D_type, int64_t bias_type, int64_t workspace_size);
    
  paddle::Tensor _counters;
  paddle::Tensor _ubuf;
  std::vector<paddle::Tensor> _ubufs;
  DType _ubuf_dtype;

  float *_ubuf_scale_inv_ptr{nullptr};
  bool _ubuf_scale_inv_initialized{false};
  int _ubuf_bytes, _ubuf_chunk_bytes;
};

} // namespace comm_gemm_overlap
} // namespace paddle_ext
} // namespace transformer_engine
