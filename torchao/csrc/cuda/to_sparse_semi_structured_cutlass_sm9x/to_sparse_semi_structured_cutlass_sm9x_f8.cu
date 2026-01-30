// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

#include "to_sparse_semi_structured_cutlass_sm9x.cuh"

namespace torchao {

std::tuple<Tensor, Tensor>
to_sparse_semi_structured_cutlass_sm9x_f8(const Tensor& W) {
  // Validate input datatypes.
  STD_TORCH_CHECK(W.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2 ||
                  W.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
              __func__, " : The input datatype ", static_cast<int>(W.scalar_type()),
              " is not supported");

#if defined(BUILD_TO_SPARSE_SEMI_STRUCTURED_CUTLASS_SM9X)
  // Dispatch to appropriate kernel template.
  switch (W.scalar_type()) {
    case torch::headeronly::ScalarType::Float8_e5m2: {
      using DtypeW = cutlass::float_e5m2_t;
      return to_sparse_semi_structured_cutlass_sm9x<DtypeW>(W);
    }
    case torch::headeronly::ScalarType::Float8_e4m3fn: {
      using DtypeW = cutlass::float_e4m3_t;
      return to_sparse_semi_structured_cutlass_sm9x<DtypeW>(W);
    }
    default:
      return std::tuple(Tensor{}, Tensor{});
  }
#else
  STD_TORCH_CHECK(false, OPERATOR_NAME, " : Not implemented");
  return std::tuple(Tensor{}, Tensor{});
#endif
}

STABLE_TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::to_sparse_semi_structured_cutlass_sm9x_f8",
         TORCH_BOX(&to_sparse_semi_structured_cutlass_sm9x_f8));
}

}  // namespace torchao
