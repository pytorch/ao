#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
#define BUILD_ROWWISE_SCALED_LINEAR_CUTLASS
#endif

#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
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
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename ThreadblockSwizzle,
    int NumStages,
    typename ElementA,
    typename ElementB,
    typename ElementOutput,
    typename ElementC,
    typename UseTensorC,
    typename ElementAScale,
    typename ElementBScale>
void rowwise_scaled_linear_kernel_cutlass_sm8x(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  static_assert((cutlass::sizeof_bits<ElementA>::value >= 8 ||
                 8 % cutlass::sizeof_bits<ElementA>::value == 0) &&
                (cutlass::sizeof_bits<ElementB>::value >= 8 ||
                 8 % cutlass::sizeof_bits<ElementB>::value == 0));

  using SmArch = cutlass::arch::Sm80;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // TODO: use FP32 if either ElementA/B is FP
  using ElementAccumulator = int32_t;
  using Operator =
      std::conditional_t<std::is_same<ElementA, ElementB>::value,
                         cutlass::arch::OpMultiplyAddSaturate,
                         cutlass::arch::OpMultiplyAddMixedInputUpcast>;

  using ElementEpilogue = float;

  constexpr auto NumEVTEpilogueStages = 1;

  const int m = tensor_a.size(0);
  const int n = tensor_b.size(0);
  int k = tensor_a.size(1);
  if constexpr (cutlass::sizeof_bits<ElementA>::value < 8) {
    k *= 8 / cutlass::sizeof_bits<ElementA>::value;
  }

  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentAScale =
      128 / cutlass::sizeof_bits<ElementAScale>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr int AlignmentBScale =
      128 / cutlass::sizeof_bits<ElementBScale>::value;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentOutput =
      128 / cutlass::sizeof_bits<ElementOutput>::value;

  // Check for current CUTLASS limitations w.r.t. alignments.
  TORCH_CHECK(k % AlignmentA == 0, OPERATOR_NAME,
              " : Number of columns of tensor A must be divisible by ",
              AlignmentA);
  TORCH_CHECK(k % AlignmentB == 0, OPERATOR_NAME,
              " : Number of columns of tensor B must be divisible by ",
              AlignmentB);
  TORCH_CHECK(n % AlignmentC == 0, OPERATOR_NAME,
              " : Number of columns of tensor C must be divisible by ",
              AlignmentC);

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          ElementOutput,
          AlignmentOutput,
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
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementC>;
  using TensorCTensor =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          OutputTileThreadMap,
          ElementC,
          cute::Stride<cute::_0, cute::_1, int64_t>>;
  using TensorC =
      std::conditional_t<UseTensorC::value, TensorCTensor, TensorCScalar>;
  using TensorCArguments = typename TensorC::Arguments;

  using ApplyAScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyAScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyAScale,
      Accum,
      TensorAScale>;

  using ApplyBScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyBScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyBScale,
      EVTApplyAScale,
      TensorBScale>;

  using ApplySum = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplySum = cutlass::epilogue::threadblock::Sm80EVT<
      ApplySum,
      EVTApplyBScale,
      TensorC>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplySum>;

  using EVTKernel = torchao::enable_2x_kernel_for_sm80_or_later<
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
          ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
          ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
          ElementOutput, LayoutOutput, AlignmentOutput,
          ElementAccumulator,
          ElementEpilogue,
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

  TensorAScaleArguments tensor_a_scale_arguments{
    (ElementAScale*)tensor_a_scale.data_ptr(),
    ElementAScale(1),
    {cute::_1{}, cute::_0{}, problem_size.m()}
  };
  TensorBScaleArguments tensor_b_scale_arguments{
    (ElementBScale*)tensor_b_scale.data_ptr(),
    ElementBScale(1),
    {cute::_0{}, cute::_1{}, problem_size.n()}
  };
  TensorCArguments tensor_c_arguments{
    [&]() -> TensorCArguments {
      if constexpr (UseTensorC::value) {
        return {(ElementC*)tensor_c.data_ptr(),
                ElementC(0),
                {cute::_0{}, cute::_1{}, problem_size.n()}};
      } else {
        return {ElementC(0)};
      }
    }()
  };
  typename Output::Arguments output_arguments{
    (ElementOutput*)tensor_d.data_ptr(),
    {problem_size.n(), cute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
    {
      {
        {
          {},                        // Accum
          tensor_a_scale_arguments,  // TensorAScale
          {}                         // ApplyAScale
        },                           // EVTApplyAScale
        tensor_b_scale_arguments,    // TensorBScale
        {},                          // ApplyBScale
      },                             // EVTApplyBScale
      tensor_c_arguments,            // TensorC
      {}                             // ApplySum
    },                               // EVTApplySum
    output_arguments                 // Output
  };                                 // EVTOutput

  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    SplitKFactor,
    callback_arguments,              // arguments of EVT callbacks
    (ElementA*)tensor_a.data_ptr(),
    (ElementB*)tensor_b.data_ptr(),
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
  auto workspace = tensor_a.new_empty({(int64_t)workspace_size},
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

template<typename ElementA, typename ElementB, typename... Types>
static void select_config(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;

  if (is_sm8x) {
    if constexpr (std::is_same<ElementA, cutlass::int4b_t>::value &&
                  std::is_same<ElementB, cutlass::int4b_t>::value) {
      using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
      using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

      // some basic tuning
      if (tensor_a.size(0) <= 16) {
        using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 256>;
        using WarpShape = cutlass::gemm::GemmShape<16, 32, 256>;
        constexpr auto NumStages = 5;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
            NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      } else if (tensor_a.size(0) <= 32) {
        using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 256>;
        using WarpShape = cutlass::gemm::GemmShape<32, 32, 256>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
            NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      } else if (tensor_a.size(0) <= 128) {
        using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 256>;
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 256>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
            NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
            tensor_d);
      } else {
        using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
            NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
            tensor_d);
      }
      return;
    } else if constexpr (std::is_same<ElementA, int8_t>::value &&
                  std::is_same<ElementB, cutlass::int4b_t>::value) {
      using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
      using ThreadblockSwizzle =
        cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

      // A minimal heuristic to improve performance for small number
      // of inputs cases.
      if (tensor_a.size(0) <= 16) {
        using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<16, 32, 128>;
        constexpr auto NumStages = 6;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
          NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      } else if (tensor_a.size(0) <= 32) {
        using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<32, 32, 128>;
        constexpr auto NumStages = 5;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
          NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      } else {
        using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
        constexpr auto NumStages = 4;
        rowwise_scaled_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, ThreadblockSwizzle,
          NumStages, ElementA, ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      }
      return;
    }
  }

  TORCH_CHECK(false, OPERATOR_NAME, " : Operator not supported on SM",
              dprops->major, ".", dprops->minor, " for given operands");
}

template<
   typename ElementA,
   typename ElementB,
   typename ElementOutput,
   typename... Types>
static void
dispatch_on_tensor_c(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  if (tensor_c.numel() == 0) {
    using ElementC = ElementOutput;
    using UseTensorC = std::false_type;
    select_config<
      ElementA, ElementB, ElementOutput, ElementC, UseTensorC, Types...>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  }

  using UseTensorC = std::true_type;
  if (tensor_c.scalar_type() == at::ScalarType::Half) {
    using ElementC = cutlass::half_t;
    select_config<
      ElementA, ElementB, ElementOutput, ElementC, UseTensorC, Types...>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  } else if (tensor_c.scalar_type() == at::ScalarType::BFloat16) {
    using ElementC = cutlass::bfloat16_t;
    select_config<
      ElementA, ElementB, ElementOutput, ElementC, UseTensorC, Types...>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  }

  TORCH_CHECK(false, OPERATOR_NAME, " : Operator not supported for datatype ",
              tensor_c.scalar_type(), " for addend");
}

template<typename ElementA, typename ElementB>
static void
dispatch_on_tensor_a_scale_and_tensor_b_scale(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  TORCH_CHECK(tensor_d.scalar_type() == tensor_a_scale.scalar_type(),
              OPERATOR_NAME, " : Operator not supported for output datatype ",
              tensor_d.scalar_type(), " as it's different from the first ",
              " operand scale datatype ", tensor_a_scale.scalar_type());

  if (tensor_a_scale.scalar_type() == at::ScalarType::Half &&
      tensor_b_scale.scalar_type() == at::ScalarType::Half) {
    using ElementAScale = cutlass::half_t;
    using ElementBScale = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    dispatch_on_tensor_c<ElementA, ElementB, ElementOutput, ElementAScale,
      ElementBScale>(
        tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c, tensor_d);
    return;
  } else if (tensor_a_scale.scalar_type() == at::ScalarType::BFloat16 &&
             tensor_b_scale.scalar_type() == at::ScalarType::BFloat16) {
    using ElementAScale = cutlass::bfloat16_t;
    using ElementBScale = cutlass::bfloat16_t;
    using ElementOutput = cutlass::bfloat16_t;
    dispatch_on_tensor_c<ElementA, ElementB, ElementOutput, ElementAScale,
      ElementBScale>(
        tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c, tensor_d);
    return;
  }

  TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for combination of data types ",
              tensor_a_scale.scalar_type(), " for first operand scale and ",
              tensor_b_scale.scalar_type(), " for second operand scale");
}

template<typename ElementA, typename ElementB>
void
rowwise_scaled_linear_cutlass_check_inputs(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias){
  // Validate layouts of arguments.
  TORCH_CHECK(xq.dim() >= 2, OPERATOR_NAME,
              " : Expected xq argument to be 2D or higher-dimensional tensor, "
              "got ", xq.dim(), " dims");
  TORCH_CHECK(xq.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected xq argument to be strided, got layout ",
              xq.layout());
  TORCH_CHECK(x_scale.dim() == xq.dim() - 1, OPERATOR_NAME,
              " : Expected xq scale argument to be ", xq.dim() - 1,
              "D tensor, got ", x_scale.dim(), " dims");
  TORCH_CHECK(x_scale.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected xq scale argument to be strided, got layout ",
              x_scale.layout());
  TORCH_CHECK(wq.dim() == 2, OPERATOR_NAME,
              " : Expected wq argument to be 2D tensor, got ", wq.dim(),
              " dims");
  TORCH_CHECK(wq.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected wq argument to be strided, got layout ",
              wq.layout());
  TORCH_CHECK(w_scale.dim() == 1 || w_scale.dim() == 2, OPERATOR_NAME,
              " : Expected wq scale argument to be 1D or 2D tensor, ", "got ",
              w_scale.dim(), " dims");
  TORCH_CHECK(w_scale.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected wq scale argument to be strided, got layout ",
              w_scale.layout());
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.dim() == 1, OPERATOR_NAME,
                " : Expected bias argument to be 1D tensor, got ", bias.dim(),
                " dims");
    TORCH_CHECK(bias.layout() == at::Layout::Strided, OPERATOR_NAME,
                " : Expected bias argument to be strided, got layout ",
                bias.layout());
  }

  // Validate sizes of arguments.
  const auto xq_sizes = xq.sizes().vec();
  TORCH_CHECK(xq_sizes.back() == wq.size(1) ||
                  xq_sizes.back() == 2 * wq.size(1),
              OPERATOR_NAME, " : Expected xq argument to have ", wq.size(1),
              " or ", 2 * wq.size(1), " columns, but got ", xq_sizes.back());
  const auto x_scale_sizes = x_scale.sizes().vec();
  for (auto i = 0; i < x_scale_sizes.size(); ++i)
    TORCH_CHECK(x_scale_sizes[i] == xq_sizes[i], OPERATOR_NAME,
                " : Expected xq scale argument size at position ", i, " to be ",
                xq_sizes[i], ", but got ", x_scale_sizes[i]);
  TORCH_CHECK(w_scale.numel() == wq.size(0), OPERATOR_NAME,
              " : Expected wq scale argument to have ", wq.size(0),
              " elements, got ", w_scale.numel(), " elements");
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.numel() == wq.size(0), OPERATOR_NAME,
                " : Expected bias argument to have ", wq.size(0),
                " elements, got ", bias.numel(), " elements");
  }

  // Validate strides of arguments.
  const auto xq_strides = xq.strides();
  TORCH_CHECK(xq_strides[xq_strides.size() - 1] == 1, OPERATOR_NAME,
              " : Expected xq argument in row-major layout");
  auto xq_stride_expected = xq_strides[xq_strides.size() - 2];
  for (int i = xq_strides.size() - 3; i >= 0; --i) {
    xq_stride_expected *= xq_sizes[i + 1];
    TORCH_CHECK(xq_strides[i] == xq_stride_expected, OPERATOR_NAME,
                " : Expected xq argument in row-major layout");
  }
  TORCH_CHECK(x_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected xq scale argument to be contiguous");
  const auto wq_strides = wq.strides();
  TORCH_CHECK(wq_strides[0] >= 1 && wq_strides[1] == 1, OPERATOR_NAME,
              " : Expected wq argument in row-major layout");
  TORCH_CHECK(w_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected wq scale argument to be contiguous");
  if (bias.numel() > 0) {
    const auto bias_strides = bias.strides();
    TORCH_CHECK(bias_strides[0] == 1, OPERATOR_NAME,
                " : Expected bias argument to be contiguous");
  }
}
#endif

// Perform linear operation, using corresponding CUTLASS datatypes
// GEMM kernel, to given arguments - result produced is:
// (tensor_a * tensor_a_scale) @ (tensor_b * tensor_b_scale).T + tensor_c
//
// Notes: The "tensor_a" and "tensor_b" are expected to be 2D tensors.
// The "tensor_a_scale" tensor is expected to be a vector, of size
// equal to number of rows of "tensor_a" tensor.  The "tensor_b_scale"
// tensor is expected to be a vector, of size equal to number of rows
// of "tensor_b" tensor. The "tensor_c" tensor is expected to be a
// vector, of size equal to number of rows of "tensor_b" tensor.
template <typename ElementA, typename ElementB>
at::Tensor
rowwise_scaled_linear_cutlass(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias) {
#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
  // Check inputs.
  rowwise_scaled_linear_cutlass_check_inputs<ElementA, ElementB>(
      xq, x_scale, wq, w_scale, bias);

  // Squash the input tensors as appropriate.
  const auto xq_sizes = xq.sizes().vec();
  const auto xq_2d = xq.reshape({-1, xq_sizes.back()});
  const auto x_scale_1d = x_scale.reshape({-1});
  const auto w_scale_1d = w_scale.reshape({-1});

  // Create result tensor.
  at::Tensor result =
      x_scale.new_empty({xq_2d.size(0), wq.size(0)});

  // Dispatch to appropriate kernel template.
  dispatch_on_tensor_a_scale_and_tensor_b_scale<ElementA, ElementB>(
      xq_2d, x_scale_1d, wq, w_scale_1d, bias, result);

  // Reshape and return result tensor.
  auto result_sizes = xq_sizes;
  result_sizes.back() = wq.size(0);
  return result.reshape(result_sizes);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return at::Tensor{};
#endif
}

}  // namespace torchao
