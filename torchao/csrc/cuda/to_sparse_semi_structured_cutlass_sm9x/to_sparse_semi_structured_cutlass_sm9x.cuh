// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <tuple>
#include <string>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda.h>  // For CUDA_VERSION
#include <cuda_runtime.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12020)
#define BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X
#endif

#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
#include <cutlass/cutlass.h>
#include <cutlass/detail/dependent_false.hpp>
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>
#include <cutlass/transform/device/transform_universal_adapter.hpp>
#include <cutlass/transform/kernel/sparse_gemm_compressor.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "cutlass_extensions/common.h"
#endif

#define OPERATOR_NAME "to_sparse_semi_structured_cutlass_sm9x"

// Macro for checking CUDA kernel launch errors (replacement for C10_CUDA_KERNEL_LAUNCH_CHECK)
#define CHECK_CUDA_KERNEL_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        STD_TORCH_CHECK(err == cudaSuccess, \
            OPERATOR_NAME, " : CUDA kernel launch failed: ", cudaGetErrorString(err)); \
    } while(0)

using torch::stable::Tensor;
namespace tsa = torch::stable::accelerator;

namespace torchao {

#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
template<typename DtypeW>
std::tuple<Tensor, Tensor>
to_sparse_semi_structured_kernel_cutlass_sm9x(const Tensor& W) {
  // The kernel doesn't check, but assumes instead, that the input
  // tensor is a structured sparse tensor.

  static_assert(std::is_same_v<DtypeW, cutlass::float_e5m2_t> ||
                std::is_same_v<DtypeW, cutlass::float_e4m3_t>);

  using SmArch = cutlass::arch::Sm90;

  using ProblemShape = cute::Shape<int, int, int, int>;

  using LayoutTagW = cutlass::layout::RowMajor;
  using StrideW = cutlass::gemm::TagToStrideA_t<LayoutTagW>;

  using DtypeMeta = unsigned char;

  // CUTLASS derives the sparse config from the mainloop.  In order
  // not to instantiate the whole mainloop here, the sparse config for
  // FP8 case is hard-coded below.  The config template arguments are
  // found by changing input data types for CUTLASS example 62 to FP8,
  // and then printing out the sparse config data type.
  using SparseConfig = cutlass::Sm90GemmSparseConfig<
    cute::sparse_elem<2, DtypeW>,
    cute::GMMA::Major::K,
    cute::sparse_elem<8, unsigned char>,
    cute::_128>;

  using CompressorUtility =
    cutlass::transform::kernel::StructuredSparseCompressorUtility<
      ProblemShape, DtypeW, LayoutTagW, SparseConfig>;
  using CompressorKernel = enable_3x_kernel_for_sm90_or_later<
    cutlass::transform::kernel::StructuredSparseCompressor<
      ProblemShape, DtypeW, LayoutTagW, SparseConfig, SmArch>>;
  using Compressor =
    cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  const int m = W.size(0);
  const int k = W.size(1);

  ProblemShape problem_shape(m, 1, k, 1);

  StrideW stride_W =
    cutlass::make_cute_packed_stride(StrideW{}, cute::make_shape(m, k, 1));
  CompressorUtility compressor_utility(problem_shape, stride_W);
  int k_compressed = compressor_utility.get_tensorA_k_physical();
  int m_meta = compressor_utility.get_metadata_m_physical();
  int k_meta = compressor_utility.get_metadata_k_physical();

  // Create result tensors.
  Tensor W_compressed = torch::stable::new_empty(W, {m, k_compressed});
  Tensor W_meta =
    torch::stable::new_empty(W, {m_meta, k_meta}, torch::headeronly::ScalarType::Byte);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
    cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
      hw_info.device_id);
  typename Compressor::Arguments arguments{
      problem_shape,
      {
        (DtypeW*)W.data_ptr(), stride_W, (DtypeW*)W_compressed.data_ptr(),
        (DtypeMeta*)W_meta.data_ptr()
      },
      {hw_info}};

  Compressor compressor_op;

  cutlass::Status status;

  // Verify that compression operation with given arguments can be
  // performed by CUTLASS.
  status = compressor_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Allocate workspace for the compressor.
  const auto workspace_size = Compressor::get_workspace_size(arguments);
  Tensor workspace = torch::stable::new_empty(W, {(int64_t)workspace_size},
                                              torch::headeronly::ScalarType::Byte);

  // Get CUDA stream from the tensor's device.
  int32_t device_idx = static_cast<int32_t>(W.get_device());
  void* stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  // Initialize compressor.
  status = compressor_op.initialize(arguments, workspace.data_ptr(), stream);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Perform compression.
  status = compressor_op.run(stream);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  CHECK_CUDA_KERNEL_LAUNCH();

  return std::make_tuple(W_compressed, W_meta);
}

template<typename DtypeW>
void
to_sparse_semi_structured_cutlass_sm9x_check_inputs(const Tensor& W) {
  // Validate the input tensor layout.
  STD_TORCH_CHECK(W.dim() == 2, OPERATOR_NAME,
              " : Expected W argument to be 2D tensor,  got ", W.dim(),
              " dims");

  // Note: layout() check removed as torch::stable::Tensor doesn't support it.
  // The stride check below implicitly validates the tensor is strided.

  // Validate the input tensor shape.
  STD_TORCH_CHECK(W.size(1) % 8 == 0, OPERATOR_NAME,
              " : Expected number of columns of the W argument to be ",
              "divisible by 8, got ", W.size(1), " columns");

  // Validate the input tensor strides.
  STD_TORCH_CHECK(W.stride(1) == 1, OPERATOR_NAME,
              " : Expected W argument in row-major layout");
}
#endif

template <typename DtypeW>
std::tuple<Tensor, Tensor>
to_sparse_semi_structured_cutlass_sm9x(const Tensor& W) {
#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
  // Get device properties using raw CUDA API.
  int device_id = W.get_device();
  cudaDeviceProp device_prop;
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_id);
  STD_TORCH_CHECK(err == cudaSuccess,
                  OPERATOR_NAME, " : cudaGetDeviceProperties failed: ",
                  cudaGetErrorString(err));
  const auto is_sm9x = device_prop.major == 9;
  if (!is_sm9x) {
    STD_TORCH_CHECK(false, OPERATOR_NAME,
                " : Operator not supported on SM", device_prop.major, ".",
                device_prop.minor, " for given operands");
  }

  // Check inputs.
  to_sparse_semi_structured_cutlass_sm9x_check_inputs<DtypeW>(W);

  // Call the kernel.
  return to_sparse_semi_structured_kernel_cutlass_sm9x<DtypeW>(W);
#else
  STD_TORCH_CHECK(false, OPERATOR_NAME, " : Not implemented");
  return std::make_tuple(Tensor{}, Tensor{});
#endif
}

}  // namespace torchao
