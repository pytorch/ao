#pragma once

#include <ATen/core/Tensor.h>

namespace torchao {

at::Tensor
rowwise_scaled_linear_sparse_cutlass_e4m3e4m3(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const at::Tensor& bias);

}  // namespace torchao
