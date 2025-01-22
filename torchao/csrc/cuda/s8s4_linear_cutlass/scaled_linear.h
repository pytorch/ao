#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                __func__, " : Got CUTLASS error: ",                       \
                cutlassGetStatusString(status));                          \
  }

namespace torchao {

template<
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages,
    typename ThreadblockSwizzle,
    typename ElementA,
    typename ElementB,
    typename ElementAccumulator,
    typename Operator,
    typename ElementAScale,
    typename ElementBScale,
    typename ElementC,
    typename UseTensorC,
    typename ElementOutput>
void scaled_linear_kernel_cutlass_sm8x(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  using SmArch = cutlass::arch::Sm80;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementEpilogue = float;
  constexpr auto NumEVTEpilogueStages = 1;

  const int m = tensor_a.size(0);
  const int n = tensor_b.size(0);
  const int k = std::is_same<ElementA, cutlass::int4b_t>::value ?
                tensor_a.size(1) * 2 :
                tensor_a.size(1);

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

  // GemmUniversalBase doesn't work with W4A4
  // using Gemm = cutlass::gemm::device::GemmUniversalBase<EVTKernel>;
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
  // constexpr auto AvailSms = -1;

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
    0
    // ,                               // stride D (unused)
    // AvailSms  // GemmUniversalBase requires passing AvailSms, but GemmUniversalAdapter doesn't
    );

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

} // namespace torchao
