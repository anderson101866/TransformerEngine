/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "comm_gemm_overlap.h"

#include <cuda_fp8.h>

#include <tuple>

#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

#include "common.h"
#include "common/util/logging.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace paddle_ext {

template <typename T>
inline void *GetOptionalDataPtr(comm_gemm_overlap::optional_tensor_ref x,  // NOLINT
                                int64_t index) {
  return x ? GetDataPtr<T>(*x, index) : nullptr;
}
inline void *GetOptionalDataPtr(comm_gemm_overlap::optional_tensor_ref x) {  // NOLINT
  return x ? x->data() : nullptr;
}
inline std::vector<size_t> GetShapeArray(const comm_gemm_overlap::optional_tensor_ref x) {
  if (x) return GetShapeArray(*x);
  return {0};
}

namespace comm_gemm_overlap {

PaddleDistributedCallbackHolder *_callback_holder = nullptr;

void set_comm_overlap_callbacks(
    PaddleDistributedCallbackHolder *callback_holder,
    const std::function<void(/*out*/ paddle::Tensor &, const paddle::Tensor &, const std::string &)> &allgather,
    const std::function<void(/*out*/ paddle::Tensor &, int64_t, const std::string &)> &bcast,
    const std::function<void(const std::string &)> &barrier) {
  _callback_holder = callback_holder;
  _callback_holder->allgather = allgather;
  _callback_holder->bcast = bcast;
  _callback_holder->barrier = barrier;
}

/*
** Python callback for torch.distributed.all_gather_into_tensor(global_data, localdata, tp_group).
*/
void ub_paddle_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                         char *group) {
  NVTE_CHECK(_callback_holder,
             "TE internal error: must set paddle.distributed callbacks during initialize_ub()");
  const auto localtensor = paddle::from_blob(
      localdata, {static_cast<int64_t>(localbytes / sizeof(uint8_t))}, paddle::DataType::UINT8,
      phi::DataLayout::NCHW,  //default layout
      phi::CPUPlace());
  auto globaltensor = paddle::from_blob(
      globaldata, {static_cast<int64_t>(globalbytes / sizeof(uint8_t))}, paddle::DataType::UINT8,
      phi::DataLayout::NCHW,  //default layout
      phi::CPUPlace());
  _callback_holder->allgather(globaltensor, localtensor, group);
  if (globaltensor.data() != globaldata) {
    NVTE_CHECK(globaltensor.is_cpu(),
               "TE internal error: gathered data should be moved to host side to perform memcpy.");
    memcpy(globaldata, globaltensor.data(), globalbytes);
  }
}

/*
** Python callback for torch.distributed.broadcast(data, src, tp_group).
*/
void ub_torch_bcast(void *data, size_t bytes, int64_t src, char *group) {
  NVTE_CHECK(_callback_holder,
             "TE internal error: must set paddle.distributed callbacks during initialize_ub()");
  auto datatensor = paddle::from_blob(data, {static_cast<int64_t>(bytes / sizeof(uint8_t))},
                                      paddle::DataType::UINT8,
                                      phi::DataLayout::NCHW,  //default layout
                                      phi::CPUPlace());
  _callback_holder->bcast(datatensor, src, group);
  if (datatensor.data() != data) {
    NVTE_CHECK(datatensor.is_cpu(),
               "TE internal error: gathered data should be moved to host side to perform memcpy.");
    memcpy(data, datatensor.data(), bytes);
  }
}

/*
** Python callback for torch.distributed.barrier(tp_group).
*/
void ub_paddle_barrier(char *group) {
  NVTE_CHECK(_callback_holder,
             "TE internal error: must set paddle.distributed callbacks during initialize_ub()");
  _callback_holder->barrier(group);
}

inline static auto get_element_size(const paddle::Tensor &x) { return phi::SizeOf(x.dtype()); }

/***************************************************************************************************
** CommGemmOverlapP2P -- Point-2-Point (ring-exchange) comm+GEMM wrappers for PyTorch
***************************************************************************************************/

CommGemmOverlapP2P::CommGemmOverlapP2P(
  const paddle::Tensor &sample, int world_rank, int world_size, int local_rank, int local_size, 
  int node_id, int num_nodes, int tp_size, int num_max_streams, int cga_size, int num_comm_sms, 
  bool set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate, bool is_reduce_scatter)
    : transformer_engine::common::CommGemmOverlapP2P(
          world_rank, world_size, local_rank, local_size, node_id, num_nodes, tp_size,
          num_max_streams, cga_size, num_comm_sms, set_sm_margin, use_ce, atomic_gemm, aggregate,
          is_reduce_scatter, &ub_paddle_allgather, &ub_torch_bcast, &ub_paddle_barrier) {
  NVTE_CHECK(!atomic_gemm,
             "atomic_gemm is not supported yet");  //TODO(anderson): tp-comm-overlap for fp8

  const auto element_size = get_element_size(sample);
  _ubuf_bytes = sample.numel() * element_size;
  _ubuf_chunk_bytes = _ubuf_bytes / _tp_size;
  if (_is_reduce_scatter) {
    // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
    // outputs for reduction at the end of the pipelining.
    _ubuf_bytes = static_cast<int>((_ubuf_bytes / _tp_size) * (_tp_size * 2 - 1));
  }
  _ubuf_dtype = (element_size == 1) ? DType::kFloat8E4M3 : Paddle2NvteDType(sample.dtype());

  void *ubuf_ptr = nullptr;
  this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, true);
  _ubuf = paddle::from_blob(ubuf_ptr,
                            {(sample.shape()[0] / _tp_size) * _num_ubuf_chunks, sample.shape()[1]},
                            sample.dtype(), sample.layout(), sample.place());
  NVTE_CHECK(ubuf_ptr);

  // Create tensor chunks for easy management
  char *ubuf_byte_ptr = reinterpret_cast<char *>(ubuf_ptr);
  const std::initializer_list<int64_t> kChunkShape{sample.shape()[0] / _tp_size, sample.shape()[1]};
  for (int i = 0; i < _num_ubuf_chunks; i++) {
    auto ubuf_chunk = paddle::from_blob(ubuf_byte_ptr, kChunkShape, sample.dtype(), sample.layout(),
                                        sample.place());
    _ubufs.push_back(std::move(ubuf_chunk));
    ubuf_byte_ptr += _ubuf_chunk_bytes;
  }

  if (_atomic_gemm) {
    std::vector<int32_t> counter(_tp_size * 2, 0);
    std::fill(counter.begin(), counter.begin() + _tp_size, 1);
    if (!_is_reduce_scatter) {
      counter[_self_chunk_id /* = 0 for AG + atomic GEMM */] = 0;
    }
    _counters = CreateFromArray(counter.data(), counter.size(), sample.place());
  }
}

void CommGemmOverlapP2P::split_overlap_ag(
  const paddle::Tensor &A, const optional_tensor_ref A_scale_inverse,
  const paddle::Tensor &B, const optional_tensor_ref B_scale_inverse,
  const optional_tensor_ref bias, /*out*/paddle::Tensor &D,     // NOLINT
  optional_tensor_ref D_scale,                                  // NOLINT
  optional_tensor_ref D_amax,                                   // NOLINT
  optional_tensor_ref pre_gelu_out, paddle::Tensor &workspace,  // NOLINT
  optional_tensor_ref B_copy,                                   // NOLINT
  int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
  int64_t D_type, int64_t bias_type, bool transa, bool transb, bool grad,
  int64_t workspace_size, bool accumulate, bool use_split_accumulator) { 
  TensorWrapper A_, B_, D_, bias_, pre_gelu_out_,
      workspace_;  //TODO(anderson): replace with Structured binding after C++17
  std::tie(A_, B_, D_, bias_, pre_gelu_out_, workspace_) = _wrap_tensors(
      A, A_scale_inverse, B, B_scale_inverse, bias, D, D_scale, D_amax, pre_gelu_out, workspace,
      A_index, B_index, D_index, A_type, B_type, D_type, bias_type, workspace_size);

  auto B_copy_ =
      MakeNvteTensor(GetOptionalDataPtr(B_copy), GetShapeArray(B_copy), Int2NvteDType(B_type),
                     nullptr, nullptr, reinterpret_cast<void *>(B_.scale_inv()));

  if (is_fp8_ubuf()) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  assert(!_ubufs.empty());
  const std::initializer_list<size_t> kChunkShape{static_cast<size_t>(_ubufs[0].shape()[0]),
                                                  static_cast<size_t>(_ubufs[0].shape()[1])};
  std::vector<TensorWrapper> ubufs_;
  ubufs_.reserve(_num_ubuf_chunks);
  for (int i = 0; i < _num_ubuf_chunks; i++) {
    ubufs_.emplace_back(  //i.e. MakeNvteTensor
        _ubufs[i].data(), kChunkShape, _ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr);
  }

  const auto stream_main =
      A.stream();  //at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  CommGemmOverlapP2P::split_gemm_overlap_ag(stream_main, A_, transa, B_, transb, bias_, D_,
                                            pre_gelu_out_, ubufs_, B_copy_, workspace_, grad,
                                            accumulate, use_split_accumulator);
  //return D;
}

void CommGemmOverlapP2P::split_overlap_rs(
  const paddle::Tensor &A, const optional_tensor_ref A_scale_inverse,
  const paddle::Tensor &B, const optional_tensor_ref B_scale_inverse,
  const optional_tensor_ref bias, /*out*/paddle::Tensor &D,    // NOLINT
  optional_tensor_ref D_scale,                                 // NOLINT
  optional_tensor_ref D_amax,                                  // NOLINT
  optional_tensor_ref pre_gelu_out, paddle::Tensor &workspace, // NOLINT
  paddle::Tensor &rs_output,                                   // NOLINT
  int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
  int64_t D_type, int64_t bias_type, bool transa, bool transb, bool grad,
  int64_t workspace_size, bool accumulate, bool use_split_accumulator) {

  TensorWrapper A_, B_, D_, bias_, pre_gelu_out_,
      workspace_;  //TODO(anderson): replace with Structured binding after C++17
  std::tie(A_, B_, D_, bias_, pre_gelu_out_, workspace_) = _wrap_tensors(
      A, A_scale_inverse, B, B_scale_inverse, bias, D, D_scale, D_amax, pre_gelu_out, workspace,
      A_index, B_index, D_index, A_type, B_type, D_type, bias_type, workspace_size);

  if (is_fp8_ubuf()) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  assert(!_ubufs.empty());
  const std::initializer_list<size_t> kChunkShape{static_cast<size_t>(_ubufs[0].shape()[0]),
                                                  static_cast<size_t>(_ubufs[0].shape()[1])};
  std::vector<TensorWrapper> ubufs_;
  ubufs_.reserve(_num_ubuf_chunks);
  for (int i = 0; i < _num_ubuf_chunks; i++) {
    ubufs_.emplace_back(  //i.e. MakeNvteTensor
        _ubufs[i].data(), kChunkShape, _ubuf_dtype, D_.amax(), D_.scale(), _ubuf_scale_inv_ptr);
  }

  const auto stream_main =
      A.stream();  //at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  transformer_engine::common::CommGemmOverlapP2P::split_gemm_overlap_rs(
      stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubufs_, workspace_, grad,
      accumulate, use_split_accumulator);

  // Reduce GEMM output chunks
  auto *reduce_buf_ptr = _ubufs[_tp_size - 1].data();
  if (is_fp8_ubuf()) {
    assert(get_element_size(rs_output) == 2);
    assert(rs_output.data());
    reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output.data(), _ubuf_scale_inv_ptr,
                                          _tp_size, _ubufs[0].numel(), (cudaStream_t)stream_main);
  } else {
    using paddle::experimental::sum; //TODO(anderson): wait `sum` to become formal API
    using paddle::experimental::assign_out_; //TODO(anderson): WARNING!! `assign_out_` does *copy*. we need some in-place operation like torch::sum_out
    paddle::Tensor reduce_buf =
        paddle::from_blob(reduce_buf_ptr, {_tp_size, _ubufs[0].shape()[0], _ubufs[0].shape()[1]},
                          _ubuf.dtype(), _ubuf.layout(), _ubuf.place());
    assign_out_(sum(reduce_buf, {0}),
                rs_output);  //rs_output = torch::sum_out(rs_output, reduce_buf, 0);
  }
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
  }

  NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
}

void CommGemmOverlapP2P::copy_input_to_ubuf(const paddle::Tensor &input, bool chunk) {
  const auto stream_main =
      input.stream();  //at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  auto ubuf = chunk ? _ubufs[_tp_id]  // Copy input to the target ubuf chunk by rank offset
                    : _ubuf;
  if (input.numel() != ubuf.numel() || get_element_size(input) != get_element_size(ubuf)) {
    NVTE_ERROR("input and ubuf size do not match!");
  }
  assert(input.data());
  NVTE_CHECK_CUDA(cudaMemcpyAsync(ubuf.data(), input.data(),
                                  input.numel() * get_element_size(input), cudaMemcpyDeviceToDevice,
                                  stream_main));
}

paddle::Tensor CommGemmOverlapP2P::get_ubuf_output(NVTE_Comm_Overlap_Type comm_type) {
  uint8_t *ubuf_wt_ptr = reinterpret_cast<uint8_t*>(_ubuf.data());
  int output_c_dim0 = _ubuf.shape()[0];
  if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
    ubuf_wt_ptr += (_ubuf.numel() * get_element_size(_ubuf) / _tp_size) * _self_chunk_id;
    output_c_dim0 /= _tp_size;
  }
  return paddle::from_blob(ubuf_wt_ptr, {output_c_dim0, _ubuf.shape()[1]}, _ubuf.dtype(),
                           _ubuf.layout(), _ubuf.place());
}

bool CommGemmOverlapP2P::is_fp8_ubuf() { return get_element_size(_ubuf) == 1; }

std::tuple<TensorWrapper/*A_*/, 
           TensorWrapper/*B_*/, 
           TensorWrapper/*D_*/, 
           TensorWrapper/*bias_*/,
           TensorWrapper/*pre_gelu_out_*/,
           TensorWrapper/*workspace_*/> CommGemmOverlapP2P::_wrap_tensors(
  const paddle::Tensor &A, const optional_tensor_ref A_scale_inverse,
  const paddle::Tensor &B, const optional_tensor_ref B_scale_inverse,
  const optional_tensor_ref bias, /*out*/paddle::Tensor &D,    // NOLINT
  optional_tensor_ref D_scale,                                 // NOLINT
  optional_tensor_ref D_amax,                                  // NOLINT
  optional_tensor_ref pre_gelu_out, paddle::Tensor &workspace, // NOLINT
    int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
    int64_t D_type, int64_t bias_type, int64_t workspace_size) {
  auto A_ = MakeNvteTensor(const_cast<void *>(A.data()), GetShapeArray(A), Int2NvteDType(A_type),
                           nullptr, nullptr,
                           const_cast<void *>(GetOptionalDataPtr<float>(A_scale_inverse, A_index)));
  auto B_ = MakeNvteTensor(const_cast<void *>(B.data()), GetShapeArray(B), Int2NvteDType(B_type),
                           nullptr, nullptr,
                           const_cast<void *>(GetOptionalDataPtr<float>(B_scale_inverse, B_index)));
  auto D_ = MakeNvteTensor(D.data(), GetShapeArray(D), Int2NvteDType(D_type),
                           GetOptionalDataPtr<float>(D_amax, D_index),
                           GetOptionalDataPtr<float>(D_scale, D_index), nullptr);
  auto bias_ = MakeNvteTensor(const_cast<void *>(GetOptionalDataPtr(bias)), GetShapeArray(bias),
                              Int2NvteDType(bias_type));

  const DType gelu_dtype =
      pre_gelu_out ? Paddle2NvteDType(pre_gelu_out->dtype()) : Int2NvteDType(D_type);
  auto pre_gelu_out_ =
      MakeNvteTensor(GetOptionalDataPtr(pre_gelu_out), GetShapeArray(pre_gelu_out), gelu_dtype);

  auto workspace_ =
      MakeNvteTensor(workspace.data(), {static_cast<size_t>(workspace_size)}, DType::kByte);
  return {std::move(A_),
          std::move(B_),
          std::move(D_),
          std::move(bias_),
          std::move(pre_gelu_out_),
          std::move(workspace_)};
}

}  // namespace comm_gemm_overlap
}  // namespace paddle_ext
}  // namespace transformer_engine