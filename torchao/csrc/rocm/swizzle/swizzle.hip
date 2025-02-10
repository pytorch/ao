#include <hip/hip_runtime.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPBlas.h>
#include <ATen/hip/tunable/GemmCommon.h>
#include <ATen/hip/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

using at::Scalar;
using at::Tensor;
using at::TensorArg;
using c10::IntArrayRef;

Tensor swizzle_stub(const Tensor& w) {
  return at::zeros_like(w);
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::swizzle_stub", &swizzle_stub);
}
