#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
#define BUILD_S4S4_LINEAR_CUTLASS
#endif

#if defined(BUILD_S4S4_LINEAR_CUTLASS)
#include "scaled_linear.h"
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#endif

namespace torchao {

#if defined(BUILD_S4S4_LINEAR_CUTLASS)

template<typename... Types>
static void select_config(
    const at::Tensor& tensor_a, const at::Tensor& tensor_a_scale,
    const at::Tensor& tensor_b, const at::Tensor& tensor_b_scale,
    const at::Tensor& tensor_c, at::Tensor& tensor_d) {
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major >= 8;

  if (is_sm8x) {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementAccumulator = int32_t;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 128>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
    constexpr auto NumStages = 3;
    using Operator = cutlass::arch::OpMultiplyAddSaturate;
    // using Operator = cutlass::arch::OpMultiplyAddMixedInputUpcast;  // this does not work
    using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

    scaled_linear_kernel_cutlass_sm8x<
      ThreadblockShape, WarpShape, InstructionShape, NumStages,
      ThreadblockSwizzle, ElementA, ElementB, ElementAccumulator, Operator,
      Types...>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  }

  TORCH_CHECK(false,
              __func__, " : Operator not supported on SM", dprops->major, ".",
              dprops->minor, " for given operands");
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
    select_config<
      ElementAScale, ElementBScale, ElementC, UseTensorC, ElementOutput>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  }

  using UseTensorC = std::true_type;
  if (tensor_c.scalar_type() == at::ScalarType::Half) {
    using ElementC = cutlass::half_t;
    select_config<
      ElementAScale, ElementBScale, ElementC, UseTensorC, ElementOutput>(
          tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, tensor_c,
          tensor_d);
    return;
  } else if (tensor_c.scalar_type() == at::ScalarType::BFloat16) {
    using ElementC = cutlass::bfloat16_t;
    select_config<
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

static void
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
  TORCH_CHECK(xq_sizes.back() == wq.size(1),
              __func__, " : Expected xq argument to have ", wq.size(1),
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
s4s4_linear_cutlass(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias) {
#if defined(BUILD_S4S4_LINEAR_CUTLASS)
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
  m.impl("torchao::s4s4_linear_cutlass", &s4s4_linear_cutlass);
}

}  // namespace torchao
