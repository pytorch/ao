// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <algorithm>
#include <optional>
#include <deque>
#include <mutex>

#include <cuda.h>

#include <torch/csrc/stable/version.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12020)
#define BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS
#endif

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>

#include "cutlass_extensions/common.h"

#endif

#define OPERATOR_NAME "rowwise_scaled_linear_sparse_cutlass"
#define PAD_TO_MULTIPLE_OF_16(x) (((x) + 15) / 16 * 16)

namespace torchao {

using torch::stable::Tensor;
namespace tsa = torch::stable::accelerator;

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)

namespace {
inline tsa::DeviceGuard make_device_guard(const Tensor& t) {
  return tsa::DeviceGuard(static_cast<tsa::DeviceIndex>(t.get_device_index()));
}

std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    device_flags.resize(device_count);
    device_properties.resize(device_count);
    return true;
  }();
}

void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

cudaDeviceProp* get_device_prop() {
  initVectors();
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDevice failed: " +
                               std::string(cudaGetErrorString(err)));
  }

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

inline cudaStream_t get_current_cuda_stream(const Tensor& t) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      static_cast<int32_t>(t.get_device_index()), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}
} // anonymous namespace

template<
    typename DtypeXq,
    typename DtypeWq,
    typename DtypeY,
    typename UseBias,
    typename DtypeBias,
    typename DtypeXScale,
    typename DtypeWScale,
    typename TileShape,
    typename ClusterShape>
void rowwise_scaled_linear_sparse_kernel_cutlass_sm9x(
  const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
  const Tensor& W_meta, const Tensor& W_scale, const Tensor& bias,
  Tensor& Y) {
  // For CUTLASS, sparsified tensor must be the first operand, thus
  // the result will be calculated as:
  //    ((Wq @ Xq.T) * W_scale * X_scale.T + bias.T).T

  using SmArch = cutlass::arch::Sm90;

  // Use CUTLASS naming conventions for naming datatypes.
  using ElementA = DtypeWq;
  using ElementB = DtypeXq;
  using ElementD = DtypeY;
  using ElementAScale = DtypeWScale;
  using ElementBScale = DtypeXScale;
  using ElementBias = DtypeBias;

  using LayoutTagA = cutlass::layout::RowMajor;
  using LayoutTagB = cutlass::layout::ColumnMajor;
  using LayoutTagD = cutlass::layout::ColumnMajor;

  constexpr auto AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr auto AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr auto AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // TODO: use different accumulator datatype if inputs are not float.
  using ElementAccumulator = float;
  using ElementCompute = float;

  using ProblemShape = cute::Shape<int, int, int, int>;

  // If FP8FastAccum not used for kernel schedule, the performance is
  // really bad; on the other side, using it doesn't seem to affect
  // the precision much - thus, sticking with it.
  using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using AScale =
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementAScale>;
  using ApplyAScale = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    Accum,
    AScale>;
  using BScale =
    cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementBScale>;
  using ApplyBScale = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    ApplyAScale,
    BScale>;
  using BiasScalar =
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementBias>;
  using BiasTensor =
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementBias>;
  using Bias = std::conditional_t<UseBias::value, BiasTensor, BiasScalar>;
  using ApplyBias = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus, ElementCompute, ElementCompute, RoundStyle>,
    ApplyBScale,
    Bias>;
  using EVT = ApplyBias;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      SmArch, cutlass::arch::OpClassSparseTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementD, LayoutTagD, AlignmentD,
      ElementD, LayoutTagD, AlignmentD,
      EpilogueSchedule,
      EVT>::CollectiveOp;
  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      SmArch, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutTagA, AlignmentA,
      ElementB, LayoutTagB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel = enable_3x_kernel_for_sm90_or_later<
    cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideE = StrideA;
  using ElementE = typename Gemm::GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

  const int m = (int)Wq.size(0);
  const int n = (int)Xq.size(0);
  const int k = (int)Xq.size(1);

  ProblemShape problem_shape(m, n, k, 1);
  const auto layout_A = SparseConfig::fill_layoutA(problem_shape);
  const auto layout_E = SparseConfig::fill_layoutE(problem_shape);
  const auto stride_B =
    cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  const auto stride_D =
    cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_shape,
    {
      (ElementA*)Wq.data_ptr(), layout_A, (ElementB*)Xq.data_ptr(), stride_B,
      (ElementE*)W_meta.data_ptr(), layout_E
    },
    {
      {},
      (ElementD*)Y.data_ptr(), stride_D, (ElementD*)Y.data_ptr(), stride_D
    }
  };

  const typename AScale::Arguments A_scale_arguments{
    (ElementAScale*)W_scale.data_ptr(),
    ElementAScale(1),
    {cute::_1{}, cute::_0{}, cute::_0{}}
  };
  const typename BScale::Arguments B_scale_arguments{
    (ElementBScale*)X_scale.data_ptr(),
    ElementBScale(1),
    {cute::_0{}, cute::_1{}, cute::_0{}}
  };
  const auto bias_arguments{
    [&]() -> typename Bias::Arguments {
      if constexpr (UseBias::value) {
        return {
          (ElementBias*)bias.data_ptr(),
          ElementBias(0),
          {cute::_1{}, cute::_0{}, cute::_0{}}
        };
      } else {
        return {{ElementBias(0)}};
      }
    }()
  };
  arguments.epilogue.thread = {
    {
      {
        {},                 // Accum
        A_scale_arguments,  // AScale
        {}                  // ApplyAScale
      },
      B_scale_arguments,    // TensorBScale
      {},                   // ApplyBScale
    },
    bias_arguments,         // Bias
    {}                      // ApplyBiass
  };

  Gemm gemm_op;

  cutlass::Status status;

  // Verify that GEMM operation with given arguments can be performed
  // by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Allocate workspace for CUTLASS mixed datatypes GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::stable::new_empty(
      Xq, {(int64_t)workspace_size},
      std::make_optional(torch::headeronly::ScalarType::Byte));

  // Get CUDA stream from tensor
  cudaStream_t stream = get_current_cuda_stream(Xq);

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(), stream);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(stream);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename DtypeXq, typename DtypeWq, typename... Types>
static void select_config(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale, const Tensor& bias,
    Tensor& Y) {
  const auto dprops = get_device_prop();
  const auto is_sm9x = dprops->major == 9;

  if (is_sm9x) {
    if constexpr ((std::is_same<DtypeXq, cutlass::float_e4m3_t>::value &&
                   std::is_same<DtypeWq, cutlass::float_e4m3_t>::value) ||
                  (std::is_same<DtypeXq, cutlass::float_e4m3_t>::value &&
                   std::is_same<DtypeWq, cutlass::float_e5m2_t>::value) ||
                  (std::is_same<DtypeXq, cutlass::float_e5m2_t>::value &&
                   std::is_same<DtypeWq, cutlass::float_e4m3_t>::value) ||
                  (std::is_same<DtypeXq, cutlass::float_e5m2_t>::value &&
                   std::is_same<DtypeWq, cutlass::float_e5m2_t>::value)) {
      const auto m = Y.size(0);
      if (m <= 64) {
        using TileShape = cute::Shape<cute::_64, cute::_32, cute::_256>;
        using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
        rowwise_scaled_linear_sparse_kernel_cutlass_sm9x<
          DtypeXq, DtypeWq, Types..., TileShape, ClusterShape>(
            Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
        return;
      } else if (m <= 128) {
        using TileShape = cute::Shape<cute::_64, cute::_128, cute::_256>;
        using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
        rowwise_scaled_linear_sparse_kernel_cutlass_sm9x<
          DtypeXq, DtypeWq, Types..., TileShape, ClusterShape>(
            Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
        return;
      } else {
        using TileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
        using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
        rowwise_scaled_linear_sparse_kernel_cutlass_sm9x<
          DtypeXq, DtypeWq, Types..., TileShape, ClusterShape>(
            Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
        return;
      }
    }
  }

  STD_TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported on SM", dprops->major, ".",
              dprops->minor, " for given operands");
}

template<typename... Types>
static void
dispatch_on_X_scale_and_W_scale(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const Tensor& bias, Tensor& Y) {
  if (X_scale.scalar_type() == torch::headeronly::ScalarType::Half &&
      W_scale.scalar_type() == torch::headeronly::ScalarType::Half) {
    using DtypeXScale = cutlass::half_t;
    using DtypeWScale = cutlass::half_t;
    select_config<Types..., DtypeXScale, DtypeWScale>(
        Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
    return;
  } else if (X_scale.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
             W_scale.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    using DtypeXScale = cutlass::bfloat16_t;
    using DtypeWScale = cutlass::bfloat16_t;
    select_config<Types..., DtypeXScale, DtypeWScale>(
        Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
    return;
  } else if (X_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
             W_scale.scalar_type() == torch::headeronly::ScalarType::Float) {
    using DtypeXScale = float;
    using DtypeWScale = float;
    select_config<Types..., DtypeXScale, DtypeWScale>(
        Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
    return;
  }

  STD_TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for combination of data types ",
              X_scale.scalar_type(), " for first operand scale and ",
              W_scale.scalar_type(), " for second operand scale");
}

template<typename DtypeXq, typename DtypeWq, typename DtypeY>
static void
dispatch_on_bias(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const std::optional<Tensor>& bias_opt, Tensor& Y) {
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    STD_TORCH_CHECK(bias.scalar_type() == Y.scalar_type(),
                OPERATOR_NAME, " : Operator not supported for bias datatype ",
                bias.scalar_type(), " as it's different from the output ",
                " datatype ", Y.scalar_type());
  }

  using DtypeBias = DtypeY;
  if (bias_opt.has_value()) {
    using UseBias = std::true_type;
    const auto bias = *bias_opt;
    dispatch_on_X_scale_and_W_scale<
      DtypeXq, DtypeWq, DtypeY, UseBias, DtypeBias>(
          Xq, X_scale, Wq, W_meta, W_scale, bias, Y);
  } else {
    using UseBias = std::false_type;
    dispatch_on_X_scale_and_W_scale<
      DtypeXq, DtypeWq, DtypeY, UseBias, DtypeBias>(
          Xq, X_scale, Wq, W_meta, W_scale, Y, Y);
  }
}

template<typename DtypeXq, typename DtypeWq>
static void
dispatch_on_Y(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const std::optional<Tensor>& bias_opt, Tensor& Y) {
  if (Y.scalar_type() == torch::headeronly::ScalarType::Half) {
    using DtypeY = cutlass::half_t;
    dispatch_on_bias<DtypeXq, DtypeWq, DtypeY>(
        Xq, X_scale, Wq, W_meta, W_scale, bias_opt, Y);
    return;
  } else if (Y.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    using DtypeY = cutlass::bfloat16_t;
    dispatch_on_bias<DtypeXq, DtypeWq, DtypeY>(
        Xq, X_scale, Wq, W_meta, W_scale, bias_opt, Y);
    return;
  }

  STD_TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for output data type ",
              Y.scalar_type());
}

template<typename DtypeXq, typename DtypeWq>
void
check_inputs(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const std::optional<Tensor>& bias_opt) {
  using torch::headeronly::Layout;

  // Validate metadata datatype.
  STD_TORCH_CHECK(W_meta.scalar_type() == torch::headeronly::ScalarType::Byte, OPERATOR_NAME,
              " : Expected Wq meta argument to be of torch.uint8 datatype got ",
              Wq.scalar_type());

  // Validate layouts of arguments.
  STD_TORCH_CHECK(Xq.dim() >= 2, OPERATOR_NAME,
              " : Expected Xq argument to be 2D or higher-dimensional tensor, "
              " got ", Xq.dim(), " dims");
  int32_t xq_layout_int;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(Xq.get(), &xq_layout_int));
  auto xq_layout = torch::stable::detail::to<Layout>(
      torch::stable::detail::from(xq_layout_int));
  STD_TORCH_CHECK(xq_layout == Layout::Strided, OPERATOR_NAME,
              " : Expected Xq argument to be strided, got layout ", xq_layout_int);
  STD_TORCH_CHECK(X_scale.dim() == Xq.dim() - 1, OPERATOR_NAME,
              " : Expected Xq scale argument to be ", Xq.dim() - 1,
              "D tensor, got ", X_scale.dim(), " dims");
  int32_t x_scale_layout_int;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(X_scale.get(), &x_scale_layout_int));
  auto x_scale_layout = torch::stable::detail::to<Layout>(
      torch::stable::detail::from(x_scale_layout_int));
  STD_TORCH_CHECK(x_scale_layout == Layout::Strided, OPERATOR_NAME,
              " : Expected Xq scale argument to be strided, got layout ", x_scale_layout_int);
  STD_TORCH_CHECK(Wq.dim() == 2, OPERATOR_NAME,
              " : Expected Wq argument to be 2D tensor, got ", Wq.dim(),
              " dims");
  int32_t wq_layout_int;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(Wq.get(), &wq_layout_int));
  auto wq_layout = torch::stable::detail::to<Layout>(
      torch::stable::detail::from(wq_layout_int));
  STD_TORCH_CHECK(wq_layout == Layout::Strided, OPERATOR_NAME,
              " : Expected Wq argument to be strided, got layout ", wq_layout_int);
  STD_TORCH_CHECK(W_meta.dim() == 2, OPERATOR_NAME,
              " : Expected Wq meta argument to be 2D tensor, got ",
              W_meta.dim(), " dims");
  int32_t w_meta_layout_int;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(W_meta.get(), &w_meta_layout_int));
  auto w_meta_layout = torch::stable::detail::to<Layout>(
      torch::stable::detail::from(w_meta_layout_int));
  STD_TORCH_CHECK(w_meta_layout == Layout::Strided, OPERATOR_NAME,
              " : Expected Wq meta argument to be strided, got layout ", w_meta_layout_int);
  STD_TORCH_CHECK(W_scale.dim() == 1 || W_scale.dim() == 2, OPERATOR_NAME,
              " : Expected Wq scale argument to be 1D or 2D tensor, ",
              "got ", W_scale.dim(), " dims");
  int32_t w_scale_layout_int;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(W_scale.get(), &w_scale_layout_int));
  auto w_scale_layout = torch::stable::detail::to<Layout>(
      torch::stable::detail::from(w_scale_layout_int));
  STD_TORCH_CHECK(w_scale_layout == Layout::Strided, OPERATOR_NAME,
              " : Expected Wq scale argument to be strided, got layout ", w_scale_layout_int);
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    STD_TORCH_CHECK(bias.dim() == 1, OPERATOR_NAME,
                " : Expected bias argument to be 1D tensor, got ", bias.dim(),
                " dims");
    int32_t bias_layout_int;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(bias.get(), &bias_layout_int));
    auto bias_layout = torch::stable::detail::to<Layout>(
        torch::stable::detail::from(bias_layout_int));
    STD_TORCH_CHECK(bias_layout == Layout::Strided, OPERATOR_NAME,
                " : Expected bias argument to be strided, got layout ", bias_layout_int);
  }

  // Validate sizes of arguments.
  const auto Xq_size_back = Xq.size(-1);
  const auto Wq_size_0 = Wq.size(0);
  const auto Wq_size_1 = Wq.size(1);
  STD_TORCH_CHECK(Xq_size_back % 32 == 0, OPERATOR_NAME,
              " : For alignment purpose, Xq argument must have number of "
              "columns divisible by ", 32, ", got ", Xq_size_back,
              " columns");
  STD_TORCH_CHECK(Wq_size_0 % 8 == 0, OPERATOR_NAME,
              " : For alignment purpose, Wq argument to have number of rows "
              "divisible by ", 8, ", but got ", Wq_size_0, " rows");
  STD_TORCH_CHECK(Xq_size_back == 2 * Wq_size_1, OPERATOR_NAME,
              " : Expected Xq argument to have ", 2 * Wq_size_1,
              " columns, but got ", Xq_size_back);
  for (int64_t i = 0; i < X_scale.dim(); ++i) {
    STD_TORCH_CHECK(X_scale.size(i) == Xq.size(i), OPERATOR_NAME,
                " : Expected Xq scale argument size at position ", i, " to be ",
                Xq.size(i), ", but got ", X_scale.size(i));
  }
  STD_TORCH_CHECK(Wq_size_1 % 8 == 0, OPERATOR_NAME,
              " : Expected Wq argument to have number of columns divisible by ",
              " 8, got ", Wq_size_1);
  // W_meta may be padded, thus expected shape calculations for this
  // tensor are as follows.
  const auto W_meta_size_0_expected = std::max((int64_t)Wq_size_0, (int64_t)64);
  const auto W_meta_size_1_expected = std::max((int64_t)PAD_TO_MULTIPLE_OF_16((int)Wq_size_1/4), (int64_t)16);
  STD_TORCH_CHECK(W_meta.size(0) == W_meta_size_0_expected, OPERATOR_NAME,
              " : Expected Wq meta argument to have ", W_meta_size_0_expected,
              " rows, got ", W_meta.size(0), " rows");
  STD_TORCH_CHECK(W_meta.size(1) == W_meta_size_1_expected, OPERATOR_NAME,
              " : Expected Wq meta argument to hold ", W_meta_size_1_expected,
              " bytes per row to encode sparsity of Wq argument, got ",
              W_meta.size(1), " bytes");
  STD_TORCH_CHECK(W_scale.numel() == Wq_size_0, OPERATOR_NAME,
              " : Expected Wq scale argument to have ", Wq_size_0,
              " elements, got ", W_scale.numel(), " elements");
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    STD_TORCH_CHECK(bias.numel() == Wq_size_0, OPERATOR_NAME,
                " : Expected bias argument to have ", Wq_size_0,
                " elements, got ", bias.numel(), " elements");
  }

  // Validate strides of arguments.
  STD_TORCH_CHECK(Xq.stride(-1) == 1, OPERATOR_NAME,
              " : Expected Xq argument in row-major layout");
  auto Xq_stride_expected = Xq.stride(-2);
  for (int64_t i = Xq.dim() - 3; i >= 0; --i) {
    Xq_stride_expected *= Xq.size(i + 1);
    STD_TORCH_CHECK(Xq.stride(i) == Xq_stride_expected, OPERATOR_NAME,
                " : Expected Xq argument in row-major layout");
  }
  STD_TORCH_CHECK(X_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected Xq scale argument to be contiguous");
  STD_TORCH_CHECK(Wq.stride(0) >= 1 && Wq.stride(1) == 1, OPERATOR_NAME,
              " : Expected Wq argument in row-major layout");
  STD_TORCH_CHECK(W_meta.stride(0) >= 1 && W_meta.stride(1) == 1, OPERATOR_NAME,
              " : Expected Wq meta argument in row-major layout");
  STD_TORCH_CHECK(W_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected Wq scale argument to be contiguous");
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    STD_TORCH_CHECK(bias.stride(0) == 1, OPERATOR_NAME,
                " : Expected bias argument to be contiguous");
  }
}
#endif

// Perform linear operation, using corresponding CUTLASS datatypes
// GEMM kernel, to given arguments - result produced is:
//     (Xq * X_scale) @ ((Wq, W_meta) * W_scale).T + bias
template <typename DtypeXq, typename DtypeWq>
Tensor
rowwise_scaled_linear_sparse_cutlass(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const std::optional<Tensor>& bias_opt,
    const std::optional<torch::headeronly::ScalarType> out_dtype_opt) {
#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
  // Check inputs.  Note that data types are checked in the
  // corresponding dispatch methods.  The limitations on data types
  // there are mostly to control the number of templates to
  // instantiate - the number of data type combinations that could be
  // supported is actually much larger.
  check_inputs<DtypeXq, DtypeWq>(Xq, X_scale, Wq, W_meta, W_scale, bias_opt);

  // Squash the input tensors as appropriate.
  const auto Xq_size_back = Xq.size(-1);
  const auto Xq_2d = torch::stable::reshape(Xq, {-1, Xq_size_back});
  const auto X_scale_1d = torch::stable::flatten(X_scale, 0, -1);
  const auto W_scale_1d = torch::stable::flatten(W_scale, 0, -1);

  // Create result tensor.
  auto out_dtype = out_dtype_opt.has_value()
                       ? *out_dtype_opt
                       : X_scale.scalar_type();
  Tensor Y = torch::stable::new_empty(
      X_scale, {Xq_2d.size(0), Wq.size(0)}, std::make_optional(out_dtype));

  // Dispatch to appropriate kernel template.
  dispatch_on_Y<DtypeXq, DtypeWq>(
      Xq_2d, X_scale_1d, Wq, W_meta, W_scale_1d, bias_opt, Y);

  // Reshape and return Y tensor.
  std::vector<int64_t> Y_sizes;
  for (int64_t i = 0; i < Xq.dim() - 1; ++i) {
    Y_sizes.push_back(Xq.size(i));
  }
  Y_sizes.push_back(Wq.size(0));
  return torch::stable::reshape(Y, Y_sizes);
#else
  STD_TORCH_CHECK(false, OPERATOR_NAME, " : Not implemented");
  return Tensor{};
#endif
}

}  // namespace torchao
