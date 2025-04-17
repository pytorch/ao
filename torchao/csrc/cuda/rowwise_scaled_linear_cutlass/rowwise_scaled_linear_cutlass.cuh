// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
#define BUILD_ROWWISE_SCALED_LINEAR_CUTLASS
#endif

#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

#include "cutlass_extensions/common.h"
#endif

#define OPERATOR_NAME "rowwise_scaled_linear_cutlass"

namespace torchao {

#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
template<
    typename DtypeXq,
    typename DtypeWq,
    typename DtypeY,
    typename UseBias,
    typename DtypeBias,
    typename DtypeXScale,
    typename DtypeWScale,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename ThreadblockSwizzle,
    int NumStages>
void rowwise_scaled_linear_kernel_cutlass_sm8x(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const at::Tensor& bias, at::Tensor& Y) {
  using SmArch = cutlass::arch::Sm80;

  // Use CUTLASS naming conventions for naming datatypes.
  using ElementA = DtypeXq;
  using ElementB = DtypeWq;
  using ElementD = DtypeY;
  using ElementAScale = DtypeXScale;
  using ElementBScale = DtypeWScale;
  using ElementBias = DtypeBias;

  static_assert((cutlass::sizeof_bits<ElementA>::value >= 8 ||
                 8 % cutlass::sizeof_bits<ElementA>::value == 0) &&
                (cutlass::sizeof_bits<ElementB>::value >= 8 ||
                 8 % cutlass::sizeof_bits<ElementB>::value == 0));

  using LayoutTagA = cutlass::layout::RowMajor;
  using LayoutTagB = cutlass::layout::ColumnMajor;
  using LayoutTagD = cutlass::layout::RowMajor;

  // TODO: use FP32 if either ElementA/ElementB is FP
  using ElementAccumulator = int32_t;
  using Operator =
      std::conditional_t<std::is_same<ElementA, ElementB>::value,
                         cutlass::arch::OpMultiplyAddSaturate,
                         cutlass::arch::OpMultiplyAddMixedInputUpcast>;

  using ElementCompute = float;

  constexpr auto NumEVTEpilogueStages = 1;

  const int m = Xq.size(0);
  const int n = Wq.size(0);
  int k = Xq.size(1);
  if constexpr (cutlass::sizeof_bits<ElementA>::value < 8) {
    k *= 8 / cutlass::sizeof_bits<ElementA>::value;
  }

  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Check for current CUTLASS limitations w.r.t. alignments.
  TORCH_CHECK(k % AlignmentA == 0, OPERATOR_NAME,
              " : Number of columns of tensor A must be divisible by ",
              AlignmentA);
  TORCH_CHECK(k % AlignmentB == 0, OPERATOR_NAME,
              " : Number of columns of tensor B must be divisible by ",
              AlignmentB);
  TORCH_CHECK(n % AlignmentD == 0, OPERATOR_NAME,
              " : Number of columns of output tensor must be divisible by ",
              AlignmentD);

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          ElementD,
          AlignmentD,
          NumEVTEpilogueStages>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using TensorAScale =
      cutlass::epilogue::threadblock::VisitorColBroadcast<
          OutputTileThreadMap,
          ElementAScale,
          cute::Stride<cute::_1, cute::_0, int64_t>>;
  using TensorAScaleArguments = typename TensorAScale::Arguments;

  using TensorBScale =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          OutputTileThreadMap,
          ElementBScale,
          cute::Stride<cute::_0, cute::_1, int64_t>>;
  using TensorBScaleArguments = typename TensorBScale::Arguments;

  using TensorCScalar =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementBias>;
  using TensorCTensor =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          OutputTileThreadMap,
          ElementBias,
          cute::Stride<cute::_0, cute::_1, int64_t>>;
  using TensorC =
      std::conditional_t<UseBias::value, TensorCTensor, TensorCScalar>;
  using TensorCArguments = typename TensorC::Arguments;

  using ApplyAScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyAScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyAScale,
      Accum,
      TensorAScale>;

  using ApplyBScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyBScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyBScale,
      EVTApplyAScale,
      TensorBScale>;

  using ApplySum = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplySum = cutlass::epilogue::threadblock::Sm80EVT<
      ApplySum,
      EVTApplyBScale,
      TensorC>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplySum>;

  using EVTKernel = torchao::enable_2x_kernel_for_sm80_or_later<
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
          ElementA, LayoutTagA, cutlass::ComplexTransform::kNone, AlignmentA,
          ElementB, LayoutTagB, cutlass::ComplexTransform::kNone, AlignmentB,
          ElementD, LayoutTagD, AlignmentD,
          ElementAccumulator,
          ElementCompute,
          cutlass::arch::OpClassTensorOp,
          SmArch,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EVTOutput,
          ThreadblockSwizzle,
          NumStages,
          Operator,
          NumEVTEpilogueStages
    >::GemmKernel>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  constexpr auto SplitKFactor = 1;

  TensorAScaleArguments X_scale_arguments{
    (ElementAScale*)X_scale.data_ptr(),
    ElementAScale(1),
    {cute::_1{}, cute::_0{}, problem_size.m()}
  };
  TensorBScaleArguments W_scale_arguments{
    (ElementBScale*)W_scale.data_ptr(),
    ElementBScale(1),
    {cute::_0{}, cute::_1{}, problem_size.n()}
  };
  TensorCArguments bias_arguments{
    [&]() -> TensorCArguments {
      if constexpr (UseBias::value) {
        return {(ElementBias*)bias.data_ptr(),
                ElementBias(0),
                {cute::_0{}, cute::_1{}, problem_size.n()}};
      } else {
        return {{ElementBias(0)}};
      }
    }()
  };
  typename Output::Arguments output_arguments{
    (ElementD*)Y.data_ptr(),
    {problem_size.n(), cute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
    {
      {
        {
          {},                 // Accum
          X_scale_arguments,  // TensorAScale
          {}                  // ApplyAScale
        },                    // EVTApplyAScale
        W_scale_arguments,    // TensorBScale
        {},                   // ApplyBScale
      },                      // EVTApplyBScale
      bias_arguments,         // TensorC
      {}                      // ApplySum
    },                        // EVTApplySum
    output_arguments          // Output
  };                          // EVTOutput

  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    SplitKFactor,
    callback_arguments,              // arguments of EVT callbacks
    (ElementA*)Xq.data_ptr(),
    (ElementB*)Wq.data_ptr(),
    nullptr,                         // ptr C (unused)
    nullptr,                         // ptr D (unused)
    problem_size.mk().product(),     // batch stride A
    problem_size.nk().product(),     // batch stride B
    0,                               // batch stride C (unused)
    0,                               // batch stride D (unused)
    problem_size.k(),                // stride A
    problem_size.k(),                // stride B
    0,                               // stride C (unused)
    0                                // stride D (unused)
  );

  Gemm gemm_op;

  cutlass::Status status;

  // Verify that GEMM operation with given arguments can be performed
  // by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Allocate workspace for CUTLASS mixed datatypes GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = Xq.new_empty({(int64_t)workspace_size},
                                at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename DtypeXq, typename DtypeWq, typename... Types>
static void select_config(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const at::Tensor& bias, at::Tensor& Y) {
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;

  if (is_sm8x) {
    if constexpr (std::is_same<DtypeXq, cutlass::int4b_t>::value &&
                  std::is_same<DtypeWq, cutlass::int4b_t>::value) {
      using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
      using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

      // some basic tuning
      if (Xq.size(0) <= 16) {
        using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 256>;
        using WarpShape = cutlass::gemm::GemmShape<16, 32, 256>;
        constexpr auto NumStages = 5;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      } else if (Xq.size(0) <= 32) {
        using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 256>;
        using WarpShape = cutlass::gemm::GemmShape<32, 32, 256>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      } else if (Xq.size(0) <= 128) {
        using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 256>;
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 256>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      } else {
        using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      }
      return;
    } else if constexpr (std::is_same<DtypeXq, int8_t>::value &&
                         std::is_same<DtypeWq, cutlass::int4b_t>::value) {
      using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
      using ThreadblockSwizzle =
        cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

      // A minimal heuristic to improve performance for small number
      // of inputs cases.
      if (Xq.size(0) <= 16) {
        using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<16, 32, 128>;
        constexpr auto NumStages = 6;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      } else if (Xq.size(0) <= 32) {
        using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<32, 32, 128>;
        constexpr auto NumStages = 5;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      } else {
        using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
            DtypeXq, DtypeWq, Types..., ThreadblockShape, WarpShape,
            InstructionShape, ThreadblockSwizzle, NumStages>(
              Xq, X_scale, Wq, W_scale, bias, Y);
      }
      return;
    }
  }

  TORCH_CHECK(false, OPERATOR_NAME, " : Operator not supported on SM",
              dprops->major, ".", dprops->minor, " for given operands");
}

template<typename... Types>
static void
dispatch_on_X_scale_and_W_scale(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const at::Tensor& bias, at::Tensor& Y) {
  if (X_scale.scalar_type() == at::ScalarType::Half &&
      W_scale.scalar_type() == at::ScalarType::Half) {
    using DtypeXScale = cutlass::half_t;
    using DtypeWScale = cutlass::half_t;
    select_config<Types..., DtypeXScale, DtypeWScale>(
        Xq, X_scale, Wq, W_scale, bias, Y);
    return;
  } else if (X_scale.scalar_type() == at::ScalarType::BFloat16 &&
             W_scale.scalar_type() == at::ScalarType::BFloat16) {
    using DtypeXScale = cutlass::bfloat16_t;
    using DtypeWScale = cutlass::bfloat16_t;
    select_config<Types..., DtypeXScale, DtypeWScale>(
        Xq, X_scale, Wq, W_scale, bias, Y);
    return;
  } else if (X_scale.scalar_type() == at::ScalarType::Float &&
             W_scale.scalar_type() == at::ScalarType::Float) {
    using DtypeXScale = float;
    using DtypeWScale = float;
    select_config<Types..., DtypeXScale, DtypeWScale>(
        Xq, X_scale, Wq, W_scale, bias, Y);
    return;
  }

  TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for combination of data types ",
              X_scale.scalar_type(), " for first operand scale and ",
              W_scale.scalar_type(), " for second operand scale");
}

template<typename DtypeXq, typename DtypeWq, typename DtypeY>
static void
dispatch_on_bias(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const std::optional<at::Tensor>& bias_opt,
    at::Tensor& Y) {
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    TORCH_CHECK(bias.scalar_type() == Y.scalar_type(),
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
          Xq, X_scale, Wq, W_scale, bias, Y);
  } else {
    using UseBias = std::false_type;
    dispatch_on_X_scale_and_W_scale<
      DtypeXq, DtypeWq, DtypeY, UseBias, DtypeBias>(
          Xq, X_scale, Wq, W_scale, Y, Y);
  }
}

template<typename DtypeXq, typename DtypeWq>
static void
dispatch_on_Y(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const std::optional<at::Tensor>& bias_opt,
    at::Tensor& Y) {
  if (Y.scalar_type() == at::ScalarType::Half) {
    using DtypeY = cutlass::half_t;
    dispatch_on_bias<DtypeXq, DtypeWq, DtypeY>(
        Xq, X_scale, Wq, W_scale, bias_opt, Y);
    return;
  } else if (Y.scalar_type() == at::ScalarType::BFloat16) {
    using DtypeY = cutlass::bfloat16_t;
    dispatch_on_bias<DtypeXq, DtypeWq, DtypeY>(
        Xq, X_scale, Wq, W_scale, bias_opt, Y);
    return;
  }

  TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for output data type ",
              Y.scalar_type());
}

template<typename DtypeXq, typename DtypeWq>
void
check_inputs(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const std::optional<at::Tensor>& bias_opt) {
  // Validate layouts of arguments.
  TORCH_CHECK(Xq.dim() >= 2, OPERATOR_NAME,
              " : Expected Xq argument to be 2D or higher-dimensional tensor, "
              "got ", Xq.dim(), " dims");
  TORCH_CHECK(Xq.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected Xq argument to be strided, got layout ",
              Xq.layout());
  TORCH_CHECK(X_scale.dim() == Xq.dim() - 1, OPERATOR_NAME,
              " : Expected Xq scale argument to be ", Xq.dim() - 1,
              "D tensor, got ", X_scale.dim(), " dims");
  TORCH_CHECK(X_scale.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected Xq scale argument to be strided, got layout ",
              X_scale.layout());
  TORCH_CHECK(Wq.dim() == 2, OPERATOR_NAME,
              " : Expected Wq argument to be 2D tensor, got ", Wq.dim(),
              " dims");
  TORCH_CHECK(Wq.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected Wq argument to be strided, got layout ",
              Wq.layout());
  TORCH_CHECK(W_scale.dim() == 1 || W_scale.dim() == 2, OPERATOR_NAME,
              " : Expected Wq scale argument to be 1D or 2D tensor, ", "got ",
              W_scale.dim(), " dims");
  TORCH_CHECK(W_scale.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected Wq scale argument to be strided, got layout ",
              W_scale.layout());
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    TORCH_CHECK(bias.dim() == 1, OPERATOR_NAME,
                " : Expected bias argument to be 1D tensor, got ", bias.dim(),
                " dims");
    TORCH_CHECK(bias.layout() == at::Layout::Strided, OPERATOR_NAME,
                " : Expected bias argument to be strided, got layout ",
                bias.layout());
  }

  // Validate sizes of arguments.
  const auto Xq_sizes = Xq.sizes().vec();
  TORCH_CHECK(Xq_sizes.back() == Wq.size(1) ||
              Xq_sizes.back() == 2 * Wq.size(1),
              OPERATOR_NAME, " : Expected Xq argument to have ", Wq.size(1),
              " or ", 2 * Wq.size(1), " columns, but got ", Xq_sizes.back());
  const auto X_scale_sizes = X_scale.sizes().vec();
  for (auto i = 0; i < X_scale_sizes.size(); ++i)
    TORCH_CHECK(X_scale_sizes[i] == Xq_sizes[i], OPERATOR_NAME,
                " : Expected Xq scale argument size at position ", i, " to be ",
                Xq_sizes[i], ", but got ", X_scale_sizes[i]);
  TORCH_CHECK(W_scale.numel() == Wq.size(0), OPERATOR_NAME,
              " : Expected Wq scale argument to have ", Wq.size(0),
              " elements, got ", W_scale.numel(), " elements");
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    TORCH_CHECK(bias.numel() == Wq.size(0), OPERATOR_NAME,
                " : Expected bias argument to have ", Wq.size(0),
                " elements, got ", bias.numel(), " elements");
  }

  // Validate strides of arguments.
  const auto Xq_strides = Xq.strides();
  TORCH_CHECK(Xq_strides[Xq_strides.size() - 1] == 1, OPERATOR_NAME,
              " : Expected Xq argument in row-major layout");
  auto Xq_stride_expected = Xq_strides[Xq_strides.size() - 2];
  for (int i = Xq_strides.size() - 3; i >= 0; --i) {
    Xq_stride_expected *= Xq_sizes[i + 1];
    TORCH_CHECK(Xq_strides[i] == Xq_stride_expected, OPERATOR_NAME,
                " : Expected Xq argument in row-major layout");
  }
  TORCH_CHECK(X_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected Xq scale argument to be contiguous");
  const auto Wq_strides = Wq.strides();
  TORCH_CHECK(Wq_strides[0] >= 1 && Wq_strides[1] == 1, OPERATOR_NAME,
              " : Expected Wq argument in row-major layout");
  TORCH_CHECK(W_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected Wq scale argument to be contiguous");
  if (bias_opt.has_value()) {
    const auto bias = *bias_opt;
    const auto bias_strides = bias.strides();
    TORCH_CHECK(bias_strides[0] == 1, OPERATOR_NAME,
                " : Expected bias argument to be contiguous");
  }
}
#endif

// Perform linear operation, using corresponding CUTLASS datatypes
// GEMM kernel, to given arguments - result produced is:
//     (Xq * X_scale) @ (Wq * W_scale).T + bias
template <typename DtypeXq, typename DtypeWq>
at::Tensor
rowwise_scaled_linear_cutlass(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale, const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::ScalarType> out_dtype_opt) {
#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
  // Check inputs.  Note that data types are checked in the
  // corresponding dispatch methods.  The limitations on data types
  // there are mostly to control the number of templates to
  // instantiate - the number of data type combinations that could be
  // supported is actually much larger.
  check_inputs<DtypeXq, DtypeWq>(Xq, X_scale, Wq, W_scale, bias_opt);

  // Squash the input tensors as appropriate.
  const auto Xq_sizes = Xq.sizes().vec();
  const auto Xq_2d = Xq.reshape({-1, Xq_sizes.back()});
  const auto X_scale_1d = X_scale.reshape({-1});
  const auto W_scale_1d = W_scale.reshape({-1});

  // Create result tensor.
  const auto options = out_dtype_opt.has_value()
                         ? X_scale.options().dtype(*out_dtype_opt)
                         : X_scale.options();
  at::Tensor Y = at::empty({Xq_2d.size(0), Wq.size(0)}, options);

  // Dispatch to appropriate kernel template.
  dispatch_on_Y<DtypeXq, DtypeWq>(
    Xq_2d, X_scale_1d, Wq, W_scale_1d, bias_opt, Y);

  // Reshape and return result tensor.
  auto Y_sizes = Xq_sizes;
  Y_sizes.back() = Wq.size(0);
  return Y.reshape(Y_sizes);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return at::Tensor{};
#endif
}

}  // namespace torchao
