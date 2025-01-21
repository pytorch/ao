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
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages,
    typename ElementA,
    typename ElementB,
    typename ElementAccumulator,
    typename Operator,
    typename ElementAScale,
    typename ElementBScale,
    typename ElementC,
    typename UseTensorC,
    typename ElementOutput>
void s8s4_linear_kernel_cutlass_sm8x(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  using SmArch = cutlass::arch::Sm80;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementEpilogue = float;

  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

  constexpr auto NumEVTEpilogueStages = 1;

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

  using EVTKernel =
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
    if constexpr (std::is_same<ElementA, int8_t>::value &&
                  std::is_same<ElementB, cutlass::int4b_t>::value) {
      using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

      // A minimal heuristic to improve performance for small number
      // of inputs cases.
      if (tensor_a.size(0) <= 16) {
        using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<16, 32, 128>;
        constexpr auto NumStages = 6;
        s8s4_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, NumStages, ElementA,
          ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      } else if (tensor_a.size(0) <= 32) {
        using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<32, 32, 128>;
        constexpr auto NumStages = 5;
        s8s4_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, NumStages, ElementA,
          ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      } else {
        using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
        constexpr auto NumStages = 4;
        s8s4_linear_kernel_cutlass_sm8x<
          ThreadblockShape, WarpShape, InstructionShape, NumStages, ElementA,
          ElementB, Types...>(
              tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
              tensor_d);
      }
      return;
    }
  }

  TORCH_CHECK(false,
              __func__, " : Operator not supported on SM", dprops->major, ".",
              dprops->minor, " for given operands");
}

template<typename... Types>
static void
dispatch_on_tensor_a_and_tensor_b(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  if (tensor_a.scalar_type() == at::ScalarType::Char) {
    if (tensor_b.scalar_type() == at::ScalarType::Char) {
      if (tensor_a.size(1) == 2 * tensor_b.size(1)) {
        using ElementA = int8_t;
        using ElementB = cutlass::int4b_t;
        using ElementAccumulator = int32_t;
        using Operator = cutlass::arch::OpMultiplyAddMixedInputUpcast;
        select_config<
          ElementA, ElementB, ElementAccumulator, Operator, Types...>(
            tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
            tensor_d);
      }
      return;
    }
  }

  TORCH_CHECK(false,
              __func__, " : Operator not supported for combination of data ",
              "types ", tensor_a.scalar_type(), " for first operand and ",
              tensor_b.scalar_type(), " for second operand");
}


template<typename ElementAScale, typename ElementBScale, typename ElementOutput>
static void
dispatch_on_tensor_c(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  if (tensor_c.numel() == 0) {
    using ElementC = ElementOutput;
    using UseTensorC = std::false_type;
    dispatch_on_tensor_a_and_tensor_b<
      ElementAScale, ElementBScale, ElementC, UseTensorC, ElementOutput>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  }

  using UseTensorC = std::true_type;
  if (tensor_c.scalar_type() == at::ScalarType::Half) {
    using ElementC = cutlass::half_t;
    dispatch_on_tensor_a_and_tensor_b<
      ElementAScale, ElementBScale, ElementC, UseTensorC, ElementOutput>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  } else if (tensor_c.scalar_type() == at::ScalarType::BFloat16) {
    using ElementC = cutlass::bfloat16_t;
    dispatch_on_tensor_a_and_tensor_b<
      ElementAScale, ElementBScale, ElementC, UseTensorC, ElementOutput>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  }

  TORCH_CHECK(false,
              __func__, " : Operator not supported for datatype ",
                tensor_c.scalar_type(), " for addend");
}

static void
dispatch_on_tensor_a_scale_and_tensor_b_scale(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  TORCH_CHECK(tensor_d.scalar_type() == tensor_a_scale.scalar_type(),
              __func__, " : Operator not supported for output datatype ",
              tensor_d.scalar_type(), " as it's different from the first ",
              " operand scale datatype ", tensor_a_scale.scalar_type());

  if (tensor_a_scale.scalar_type() == at::ScalarType::Half &&
      tensor_b_scale.scalar_type() == at::ScalarType::Half) {
    using ElementAScale = cutlass::half_t;
    using ElementBScale = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    dispatch_on_tensor_c<ElementAScale, ElementBScale, ElementOutput>(
        tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c, tensor_d);
    return;
  } else if (tensor_a_scale.scalar_type() == at::ScalarType::BFloat16 &&
             tensor_b_scale.scalar_type() == at::ScalarType::BFloat16) {
    using ElementAScale = cutlass::bfloat16_t;
    using ElementBScale = cutlass::bfloat16_t;
    using ElementOutput = cutlass::bfloat16_t;
    dispatch_on_tensor_c<ElementAScale, ElementBScale, ElementOutput>(
        tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c, tensor_d);
    return;
  }

  TORCH_CHECK(false,
              __func__, " : Operator not supported for combination of data ",
              "types ", tensor_a_scale.scalar_type(),
              " for first operand scale and ", tensor_b_scale.scalar_type(),
                " for second operand scale");
}

void
check_inputs(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias) {
  // Validate layouts of arguments.
  TORCH_CHECK(xq.dim() >= 2,
              __func__, " : Expected xq argument to be 2D or "
              "higher-dimensional tensor, got ", xq.dim(), " dims");
  TORCH_CHECK(xq.layout() == at::Layout::Strided,
              __func__, " : Expected xq argument to be strided, got layout ",
              xq.layout());
  TORCH_CHECK(x_scale.dim() == xq.dim() - 1,
              __func__, " : Expected xq scale argument to be ", xq.dim() - 1,
              "D tensor, got ", x_scale.dim(), " dims");
  TORCH_CHECK(x_scale.layout() == at::Layout::Strided,
              __func__, " : Expected xq scale argument to be strided, got "
              "layout ", x_scale.layout());
  TORCH_CHECK(wq.dim() == 2,
              __func__, " : Expected wq argument to be 2D tensor, got ",
              wq.dim(), " dims");
  TORCH_CHECK(wq.layout() == at::Layout::Strided,
              __func__, " : Expected wq argument to be strided, got layout ",
              wq.layout());
  TORCH_CHECK(w_scale.dim() == 1 || w_scale.dim() == 2,
              __func__, " : Expected wq scale argument to be 1D or 2D tensor, ",
              "got ", w_scale.dim(), " dims");
  TORCH_CHECK(w_scale.layout() == at::Layout::Strided,
              __func__, " : Expected wq scale argument to be strided, got "
              "layout ", w_scale.layout());
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.dim() == 1,
                __func__, " : Expected bias argument to be 1D tensor, got ",
                bias.dim(), " dims");
    TORCH_CHECK(bias.layout() == at::Layout::Strided,
                __func__, " : Expected bias argument to be strided, got ",
                "layout ", bias.layout());
  }

  // Validate sizes of arguments.
  const auto xq_sizes = xq.sizes().vec();
  TORCH_CHECK(xq_sizes.back() == 2 * wq.size(1),
              __func__, " : Expected xq argument to have ", 2 * wq.size(1),
              " columns, but got ", xq_sizes.back());
  const auto x_scale_sizes = x_scale.sizes().vec();
  for (auto i = 0; i < x_scale_sizes.size(); ++i)
    TORCH_CHECK(x_scale_sizes[i] == xq_sizes[i],
                __func__, " : Expected xq scale argument size at position ",
                i, " to be ", xq_sizes[i], ", but got ", x_scale_sizes[i]);
  TORCH_CHECK(w_scale.numel() == wq.size(0),
              __func__, " : Expected wq scale argument to have ", wq.size(0),
              " elements, got ", w_scale.numel(), " elements");
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.numel() == wq.size(0),
                __func__, " : Expected bias argument to have ", wq.size(0),
                " elements, got ", bias.numel(), " elements");
  }

  // Validate strides of arguments.
  const auto xq_strides = xq.strides();
  TORCH_CHECK(xq_strides[xq_strides.size() - 1] == 1,
              __func__, " : Expected xq argument in row-major layout");
  auto xq_stride_expected = xq_strides[xq_strides.size() - 2];
  for (int i = xq_strides.size() - 3; i >= 0; --i) {
    xq_stride_expected *= xq_sizes[i + 1];
    TORCH_CHECK(xq_strides[i] == xq_stride_expected,
                __func__, " : Expected xq argument in row-major layout");
  }
  TORCH_CHECK(x_scale.is_contiguous(),
              __func__, " : Expected xq scale argument to be contiguous");
  const auto wq_strides = wq.strides();
  TORCH_CHECK(wq_strides[0] >= 1 && wq_strides[1] == 1,
              __func__, " : Expected wq argument in row-major layout");
  TORCH_CHECK(w_scale.is_contiguous(),
              __func__, " : Expected wq scale argument to be contiguous");
  if (bias.numel() > 0) {
    const auto bias_strides = bias.strides();
    TORCH_CHECK(bias_strides[0] == 1,
                __func__, " : Expected bias argument to be contiguous");
  }
}
#endif

// Perform linear operation, using corresponding CUTLASS mixed
// data-types GEMM kernel, to given arguments:
//   result = (xq * x_scale) @ (wq * w_scale).T + bias
// Notes: The "x_scale" tensor is expected to be a vector, of size
// equal to number of rows of "xq" tensor.  The "w_scale" tensor is
// expected to be a vector, of size equal to number of rows of "wq"
// tensor. The "bias" tensor is expected to be a vector, of size equal
// to number of rows of "wq" tensor.
at::Tensor
s8s4_linear_cutlass(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias) {
#if defined(BUILD_S8S4_LINEAR_CUTLASS)
  // Check inputs.
  check_inputs(xq, x_scale, wq, w_scale, bias);

  // Squash the input tensors as appropriate.
  const auto xq_sizes = xq.sizes().vec();
  const auto xq_2d = xq.reshape({-1, xq_sizes.back()});
  const auto x_scale_sizes = x_scale.sizes().vec();
  const auto x_scale_1d = x_scale.reshape({-1});
  const auto w_scale_1d = w_scale.reshape({-1});

  // Introduce alias names for arguments, according to the CUTLASS
  // naming conventions.
  const auto& tensor_a = xq_2d;
  const auto& tensor_a_scale = x_scale_1d;
  const auto& tensor_b = wq;
  const auto& tensor_b_scale = w_scale_1d;
  const auto& tensor_c = bias;

  // Create output tensor.
  at::Tensor tensor_d =
      tensor_a_scale.new_empty({tensor_a.size(0), tensor_b.size(0)});

  // Dispatch to appropriate kernel template.
  dispatch_on_tensor_a_scale_and_tensor_b_scale(
      tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c, tensor_d);

  // Reshape and return output tensor.
  auto tensor_d_sizes = xq_sizes;
  tensor_d_sizes.back() = wq.size(0);
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
