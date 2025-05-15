// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <torchao/experimental/kernels/mps/src/lowbit.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::backends::mps::delegate::getMTLBufferStorage;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::tensor_is_rank;

namespace {

std::string scalar_type_to_string(const ScalarType& scalar_type) {
  switch (scalar_type) {
    case ScalarType::Float:
      return "float";
    case ScalarType::Half:
      return "half";
    case ScalarType::BFloat16:
      return "bfloat";
    default:
      ET_CHECK_MSG(
          false, "Unsupported type by lowbit quantized linear");
      return "undefined";
  }
}

template <int nbit>
bool check_linear_mps_args(
    const Tensor& A,
    const Tensor& B,
    int64_t group_size,
    const Tensor& S,
    const Tensor& Z) {
  auto N = B.size(0);
  auto K = A.size(1);

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      A.scalar_type() == ScalarType::BFloat16 ||
          A.scalar_type() == ScalarType::Half ||
          A.scalar_type() == ScalarType::Float,
      "Expect A to be either 32-bit or 16-bit float tensor.");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      tensor_is_rank(A, 2), "Expect A to be 2D tensor.");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      B.scalar_type() == ScalarType::Byte, "Expect B to be uint8 tensor.");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      B.size(1) == (K / 8) * nbit, "Expect B.size(1) == (K / 8) * nbit");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(K % 8 == 0, "Expect K to be multiple of 8");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(N % 4 == 0, "Expect N to be multiple of 4");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      group_size == 32 || group_size == 64 || group_size == 128 ||
          group_size == 256,
      "Expect group_size to be 32, 64, 128 or 256");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      S.dim() == 2 && S.size(0) == N,
      "Expect S to be 2d tensor with shape [N, :]");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      Z.dim() == 2 && Z.size(0) == N,
      "Expect Z to be 2d tensor with shape [N, :]");

  return true;
}

template <int nbit>
Tensor& linear_mps_kernel_et_ctx_out(
    KernelRuntimeContext& ctx,
    const Tensor& A,
    const Tensor& B,
    int64_t group_size,
    const Tensor& S,
    const Tensor& Z,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_linear_mps_args<nbit>(A, B, group_size, S, Z),
      InvalidArgument,
      out);

  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  torchao::kernels::mps::lowbit::LowBitQuantWeights<nbit>::linear(
      getMTLBufferStorage(A),
      getMTLBufferStorage(B),
      group_size,
      getMTLBufferStorage(S),
      getMTLBufferStorage(Z),
      getMTLBufferStorage(out),
      M,
      K,
      N,
      scalar_type_to_string(A.scalar_type()));

  return out;
}

} // namespace

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_1bit_weight.out", linear_mps_kernel_et_ctx_out<1>);
}

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_2bit_weight.out", linear_mps_kernel_et_ctx_out<2>);
}

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_3bit_weight.out", linear_mps_kernel_et_ctx_out<3>);
}

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_4bit_weight.out", linear_mps_kernel_et_ctx_out<4>);
}

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_5bit_weight.out", linear_mps_kernel_et_ctx_out<5>);
}

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_6bit_weight.out", linear_mps_kernel_et_ctx_out<6>);
}

namespace {
EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_7bit_weight.out", linear_mps_kernel_et_ctx_out<7>);
}
