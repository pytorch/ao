#include "int4mv_kernel.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/mps/OperationUtils.h>

namespace torchao {
using Tensor = at::Tensor;

Tensor& int4mv_out_impl(const Tensor& A, const Tensor& B, int64_t groupSize, const Tensor& scalesAndZeros, Tensor& C) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == at::kBFloat16 || A.dtype() == at::kHalf || A.dtype() == at::kFloat,
              __func__,
              " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == at::kChar, __func__, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

  TORCH_CHECK(scalesAndZeros.dim() == 3 && scalesAndZeros.size(1) == N && scalesAndZeros.size(2) == 2,
              __func__,
              ": expect scalesAndZeros to be 3d tensor with sizes [:, ",
              N,
              ", 2]");

  TORCH_CHECK(N % 32 == 0 && K % 32 == 0);
  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), 0};
  std::array<uint64_t, 4> nbytes = {A.nbytes(), B.nbytes(), scalesAndZeros.nbytes(), C.nbytes()};
  std::string A_scalar_type = at::native::mps::scalarToMetalTypeString(A.scalar_type());
  int4mv(A.const_data_ptr<uint8_t>(), B.const_data_ptr<uint8_t>(), groupSize, scalesAndZeros.const_data_ptr<uint8_t>(), C.mutable_data_ptr<uint8_t>(), sizes, nbytes, A_scalar_type);
  return C;
}

Tensor int4mv_impl(const Tensor& A, const Tensor& B, int64_t groupSize, const Tensor& scalesAndZeros) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto C = at::empty({M, N}, A.options());
  return int4mv_out_impl(A, B, groupSize, scalesAndZeros, C);
}


TORCH_LIBRARY_IMPL(torchao, MPS, m) {
  m.impl("torchao::int4mv", &int4mv_impl);
  m.impl("torchao::int4mv.out", &int4mv_out_impl);
}

} // namespace torchao