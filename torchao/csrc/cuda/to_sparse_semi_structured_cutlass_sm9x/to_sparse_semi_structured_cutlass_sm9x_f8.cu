#include <torch/library.h>

#include "to_sparse_semi_structured_cutlass_sm9x.cuh"

namespace torchao {

std::tuple<at::Tensor, at::Tensor>
to_sparse_semi_structured_cutlass_sm9x_f8(const at::Tensor& W) {
  // Validate input datatypes.
  TORCH_CHECK(W.dtype() == at::kFloat8_e5m2 || W.dtype() == at::kFloat8_e4m3fn,
              __func__, " : The input datatype ", W.dtype(),
              " is not supported");

#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
  // Dispatch to appropriate kernel template.
  if (W.dtype() == at::kFloat8_e5m2) {
    using DtypeW = cutlass::float_e5m2_t;
    return to_sparse_semi_structured_cutlass_sm9x<DtypeW>(W);
  } else if (W.dtype() == at::kFloat8_e4m3fn) {
    using DtypeW = cutlass::float_e4m3_t;
    return to_sparse_semi_structured_cutlass_sm9x<DtypeW>(W);
  }
  return std::tuple(at::Tensor{}, at::Tensor{});
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return std::tuple(at::Tensor{}, at::Tensor{});
#endif
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::to_sparse_semi_structured_cutlass_sm9x_f8",
         &to_sparse_semi_structured_cutlass_sm9x_f8);
}

}  // namespace torchao
