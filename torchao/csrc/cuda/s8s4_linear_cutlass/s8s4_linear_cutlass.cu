#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
#define BUILD_S8S4_LINEAR_CUTLASS
#endif

#if defined(BUILD_S8S4_LINEAR_CUTLASS)
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                __func__, " : Got CUTLASS error: ",                       \
                cutlassGetStatusString(status));                          \
  }
#endif

namespace torchao {

#if defined(BUILD_S8S4_LINEAR_CUTLASS)
template<
    typename ElementA,
    typename ElementAScale,
    typename ElementB,
    typename ElementBScale,
    typename ElementC,
    typename ElementAccumulator,
    typename ElementEpilogue,
    typename ElementOutput,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages,
    bool use_tensor_c>
void s8s4_linear_kernel_cutlass(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const int m = tensor_a.size(0);
  const int n = tensor_b.size(0);
  const int k = tensor_a.size(1);

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
  TORCH_CHECK(k % AlignmentA == 0,
              __func__, " : Number of columns of tensor A must be divisible ",
              "by ", AlignmentA);
  TORCH_CHECK(k % AlignmentB == 0,
              __func__, " : Number of columns of tensor B must be divisible ",
              "by ", AlignmentB);
  TORCH_CHECK(n % AlignmentC == 0,
              __func__, " : Number of columns of tensor C must be divisible ",
              "by ", AlignmentC);

  using SmArch = cutlass::arch::Sm80;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

  constexpr auto NumEVTEpilogueStages = 1;

  using TensorAScaleTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          ElementAScale,
          AlignmentAScale,
          NumEVTEpilogueStages>;
  using TensorBScaleTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          ElementBScale,
          AlignmentBScale,
          NumEVTEpilogueStages>;
  using TensorCTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          ElementC,
          AlignmentC,
          NumEVTEpilogueStages>;
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
          TensorAScaleTileThreadMap,
          ElementAScale,
          cute::Stride<cute::_1, cute::_0, int64_t>>;
  using TensorAScaleArguments = typename TensorAScale::Arguments;

  using TensorBScale =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          TensorBScaleTileThreadMap,
          ElementBScale,
          cute::Stride<cute::_0, cute::_1, int64_t>>;
  using TensorBScaleArguments = typename TensorBScale::Arguments;

  using TensorCScalar =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementC>;
  using TensorCTensor =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<
          TensorCTileThreadMap,
          ElementC,
          cute::Stride<cute::_0, cute::_1, int32_t>>;
  using TensorC =
      std::conditional_t<use_tensor_c, TensorCTensor, TensorCScalar>;
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

  using EVTKernel =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
      ElementC, LayoutC, AlignmentC,
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
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      NumEVTEpilogueStages
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<EVTKernel>;

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
      if constexpr (use_tensor_c) {
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
  constexpr auto AvailSms = -1;

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
    0,                               // stride D (unused)
    AvailSms);

  Gemm gemm_op;

  cutlass::Status status;

  // Verify that GEMM operation with given arguments can be performed
  // by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status);

  // Allocate workspace for CUTLASS mixed datatypes GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = tensor_a.new_empty({(int64_t)workspace_size},
                                      at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);
}

template<
    typename ElementA,
    typename ElementAScale,
    typename ElementB,
    typename ElementBScale,
    typename ElementC,
    typename ElementAccumulator,
    typename ElementEpilogue,
    typename ElementOutput,
    bool use_tensor_c>
void
s8s4_linear_cutlass_dispatch_shapes(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  // A minimal heuristic to improve performance for small number of
  // inputs cases.
  if (tensor_a.size(0) <= 16) {
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 128>;
    constexpr auto NumStages = 6;
    s8s4_linear_kernel_cutlass<
        ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
        ElementAccumulator, ElementEpilogue, ElementOutput,
        ThreadblockShape, WarpShape, InstructionShape, NumStages, use_tensor_c>(
            tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
            tensor_d);
  } else if (tensor_a.size(0) <= 32) {
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 128>;
    constexpr auto NumStages = 5;
    s8s4_linear_kernel_cutlass<
        ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
        ElementAccumulator, ElementEpilogue, ElementOutput,
        ThreadblockShape, WarpShape, InstructionShape, NumStages, use_tensor_c>(
            tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
            tensor_d);
  } else {
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
    constexpr auto NumStages = 4;
    s8s4_linear_kernel_cutlass<
        ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
        ElementAccumulator, ElementEpilogue, ElementOutput,
        ThreadblockShape, WarpShape, InstructionShape, NumStages, use_tensor_c>(
            tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
            tensor_d);
  }
}

#endif

// Perform linear operation, using corresponding CUTLASS mixed
// data-types GEMM kernel, to given arguments:
//   result = (input * input_scale) @ (weight * weight_scale).T + bias
// Notes: The "input_scale" tensor is expected to be a vector, of size
// equal to number of rows of "input" tensor.  The "weight_scale"
// tensor is expected to be a vector, of size equal to number of rows
// of "weight" tensor. The "bias" tensor is expected to be a vector,
// of size equal to number of rows of "weight" tensor.
at::Tensor
s8s4_linear_cutlass(const at::Tensor& input, const at::Tensor& input_scale,
                    const at::Tensor& weight, const at::Tensor& weight_scale,
                    const at::Tensor& bias) {
#if defined(BUILD_S8S4_LINEAR_CUTLASS)
  // For now, only CC 8.x devices are supported.
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;
  TORCH_CHECK(is_sm8x,
              __func__, " : Supported only on GPUs with compute capability "
              "8.x");

  // Validate datatypes of arguments.
  TORCH_CHECK(input.dtype() == at::kChar,
              __func__, " : The input datatype ", input.dtype(),
              " not supported");
  TORCH_CHECK(input_scale.dtype() == at::kHalf ||
              input_scale.dtype() == at::kBFloat16,
              __func__, " : The input scale datatype ", input_scale.dtype(),
              " not supported");
  TORCH_CHECK(weight.dtype() == at::kChar, " : The weight datatype ",
              weight.dtype(), " not supported");
  TORCH_CHECK(weight_scale.dtype() == input_scale.dtype(),
              __func__, " : Expected weight scale datatype ",
              input_scale.dtype(), ", got ", weight_scale.dtype());
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.dtype() == input_scale.dtype(),
                __func__, " : Expected bias datatype ", input_scale.dtype(),
                ", got ", bias.dtype());
  }

  // Validate layouts of arguments.
  TORCH_CHECK(input.dim() >= 2,
              __func__, " : Expected input argument to be 2D or "
              "higher-dimensional tensor, got ", input.dim(), " dims");
  TORCH_CHECK(input.layout() == at::Layout::Strided,
              __func__, " : Expected input argument to be strided, got layout ",
              input.layout());
  TORCH_CHECK(input_scale.dim() == input.dim() - 1,
              __func__, " : Expected input scale argument to be ",
              input.dim() - 1, "D tensor, got ", input_scale.dim(), " dims");
  TORCH_CHECK(input_scale.layout() == at::Layout::Strided,
              __func__, " : Expected input scale argument to be strided, got "
              "layout ", input_scale.layout());
  TORCH_CHECK(weight.dim() == 2,
              __func__, " : Expected weight argument to be 2D tensor, got ",
              weight.dim(), " dims");
  TORCH_CHECK(weight.layout() == at::Layout::Strided,
              __func__,
              " : Expected weight argument to be strided, got layout ",
              weight.layout());
  TORCH_CHECK(weight_scale.dim() == 1 || weight_scale.dim() == 2,
              __func__, " : Expected weight scale argument to be 1D or 2D ",
              "tensor, got ", weight_scale.dim(), " dims");
  TORCH_CHECK(weight_scale.layout() == at::Layout::Strided,
              __func__, " : Expected weight scale argument to be strided, got "
              "layout ", weight_scale.layout());
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.dim() == 1,
                __func__, " : Expected bias argument to be 1D tensor, got ",
                bias.dim(), " dims");
    TORCH_CHECK(bias.layout() == at::Layout::Strided,
                __func__, " : Expected bias argument to be strided, got ",
                "layout ", bias.layout());
  }

  // Squash the input tensor to 2D tensor.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});
  const auto input_scale_sizes = input_scale.sizes().vec();
  const auto input_scale_1d = input_scale.reshape({-1});
  const auto weight_scale_1d = weight_scale.reshape({-1});

  // Validate sizes of arguments.
  TORCH_CHECK(input_2d.size(1) == 2 * weight.size(1),
              __func__, " : Expected input argument to have ",
              2 * weight.size(1), " columns, but got ", input_2d.size(1));
  for (auto i = 0; i < input_scale_sizes.size(); ++i)
    TORCH_CHECK(input_scale_sizes[i] == input_sizes[i],
                __func__, " : Expected input scale argument size at position ",
                i, " to be ", input_sizes[i], ", but got ",
                input_scale_sizes[i]);
  TORCH_CHECK(weight_scale_1d.numel() == weight.size(0),
              __func__, " : Expected weight scale argument to have ",
              weight.size(0), " elements, got ", weight_scale_1d.numel(),
              " elements");
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.numel() == weight.size(0),
                __func__, " : Expected bias argument to have ", weight.size(0),
                " elements, got ", bias.numel(), " elements");
  }

  // Validate strides of arguments.
  const auto input_2d_strides = input_2d.strides();
  TORCH_CHECK(input_2d_strides[0] >= 1 && input_2d_strides[1] == 1,
              __func__, " : Expected input argument in row-major layout");
  const auto input_scale_1d_strides = input_scale_1d.strides();
  TORCH_CHECK(input_scale_1d_strides[0] == 1,
              __func__, " : Expected input scale argument to be contiguous");
  const auto weight_strides = weight.strides();
  TORCH_CHECK(weight_strides[0] >= 1 && weight_strides[1] == 1,
              __func__, " : Expected weight argument in row-major layout");
  const auto weight_scale_1d_strides = weight_scale_1d.strides();
  TORCH_CHECK(weight_scale_1d_strides[0] == 1,
              __func__, " : Expected weight scale argument to be contiguous");
  if (bias.numel() > 0) {
    const auto bias_strides = bias.strides();
    TORCH_CHECK(bias_strides[0] == 1,
                __func__, " : Expected bias argument to be contiguous");
  }

  // Introduce alias names for arguments, according to the CUTLASS
  // naming conventions.
  const auto& tensor_a = input_2d;
  const auto& tensor_a_scale = input_scale_1d;
  const auto& tensor_b = weight;
  const auto& tensor_b_scale = weight_scale_1d;
  const auto& tensor_c = bias;

  // Create output tensor.
  at::Tensor tensor_d =
      tensor_a_scale.new_empty({tensor_a.size(0), tensor_b.size(0)});

  using ElementA = int8_t;
  using ElementB = cutlass::int4b_t;
  using ElementAccumulator = int32_t;
  AT_DISPATCH_SWITCH(
    input_scale.scalar_type(),
    "s8s4_linear_cutlass",
    AT_DISPATCH_CASE(
      at::ScalarType::Half,
      [&]() {
        using ElementAScale = cutlass::half_t;
        using ElementBScale = cutlass::half_t;
        using ElementC = cutlass::half_t;
        using ElementEpilogue = float;
        using ElementOutput = cutlass::half_t;
        if (bias.numel() > 0) {
          s8s4_linear_cutlass_dispatch_shapes<
              ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
              ElementAccumulator, ElementEpilogue, ElementOutput, true>(
                  tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
                  tensor_d);
        } else {
          s8s4_linear_cutlass_dispatch_shapes<
              ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
              ElementAccumulator, ElementEpilogue, ElementOutput, false>(
                  tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
                  tensor_d);
        }
      })
    AT_DISPATCH_CASE(
      at::ScalarType::BFloat16,
      [&]() {
        using ElementAScale = cutlass::bfloat16_t;
        using ElementBScale = cutlass::bfloat16_t;
        using ElementC = cutlass::bfloat16_t;
        using ElementEpilogue = float;
        using ElementOutput = cutlass::bfloat16_t;
        if (bias.numel() > 0) {
          s8s4_linear_cutlass_dispatch_shapes<
              ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
              ElementAccumulator, ElementEpilogue, ElementOutput, true>(
                  tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
                  tensor_d);
        } else {
          s8s4_linear_cutlass_dispatch_shapes<
              ElementA, ElementAScale, ElementB, ElementBScale, ElementC,
              ElementAccumulator, ElementEpilogue, ElementOutput, false>(
                  tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
                  tensor_d);
        }
      }));

  auto tensor_d_sizes = input_sizes;
  tensor_d_sizes.back() = weight.size(0);
  return tensor_d.reshape(tensor_d_sizes);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, __func__);
  return at::Tensor{};
#endif
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::s8s4_linear_cutlass", &s8s4_linear_cutlass);
}

}  // namespace torchao
