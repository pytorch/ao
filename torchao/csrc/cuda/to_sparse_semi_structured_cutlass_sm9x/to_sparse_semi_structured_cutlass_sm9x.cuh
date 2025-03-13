#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12020)
#define BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X
#endif

#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/dependent_false.hpp>
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>
#include <cutlass/transform/device/transform_universal_adapter.hpp>
#include <cutlass/transform/kernel/sparse_gemm_compressor.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "cutlass_extensions/common.h"
#endif

#define OPERATOR_NAME "to_sparse_semi_structured_cutlass_sm9x"

namespace torchao {

#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
template<typename DtypeW>
std::tuple<at::Tensor, at::Tensor>
to_sparse_semi_structured_kernel_cutlass_sm9x(const at::Tensor& W) {
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
  at::Tensor W_compressed = W.new_empty({m, k_compressed});
  at::Tensor W_meta =
    W.new_empty({m_meta, k_meta}, at::TensorOptions().dtype(at::kByte));

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
  auto workspace = W.new_empty({(int64_t)workspace_size},
                               at::TensorOptions().dtype(at::kByte));

  // Initialize compressor.
  status = compressor_op.initialize(arguments, workspace.data_ptr(),
                                    at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Perform compression.
  status = compressor_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(W_compressed, W_meta);
}
  
template<typename DtypeW>
void
to_sparse_semi_structured_cutlass_sm9x_check_inputs(const at::Tensor& W) {
  // Validate the input tensor layout.
  TORCH_CHECK(W.dim() == 2, OPERATOR_NAME,
              " : Expected W argument to be 2D tensor,  got ", W.dim(),
              " dims");
  TORCH_CHECK(W.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected W argument to be strided, got layout ",W.layout());

  // Validate the input tensor shape.
  const auto W_sizes = W.sizes().vec();
  TORCH_CHECK(W_sizes[1] % 8 == 0, OPERATOR_NAME,
              " : Expected number of columns of the W argument to be ",
              "divisible by 8, got ", W_sizes[1], " columns");

  // Validate the input tensor strides.
  const auto W_strides = W.strides();
  TORCH_CHECK(W_strides[1] == 1, OPERATOR_NAME,
              " : Expected W argument in row-major layout");
}
#endif

template <typename DtypeW>
std::tuple<at::Tensor, at::Tensor>
to_sparse_semi_structured_cutlass_sm9x(const at::Tensor& W) {
#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm9x = dprops->major == 9;
  if (!is_sm9x) {
    TORCH_CHECK(false, OPERATOR_NAME,
                " : Operator not supported on SM", dprops->major, ".",
                dprops->minor, " for given operands");
  }

  // Check inputs.
  to_sparse_semi_structured_cutlass_sm9x_check_inputs<DtypeW>(W);

  // Call the kernel.
  return to_sparse_semi_structured_kernel_cutlass_sm9x<DtypeW>(W);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return std::make_tuple(at::Tensor{}, at::Tensor{});
#endif
}
  
}  // namespace torchao
