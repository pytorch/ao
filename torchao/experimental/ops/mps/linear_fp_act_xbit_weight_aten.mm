// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off
#include <torch/library.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torchao/experimental/kernels/mps/src/lowbit.h>
// clang-format on

namespace torchao::kernels::mps::lowbit::aten {

using Tensor = at::Tensor;
using namespace at::native::mps;

// LowBit Quantized Linear on MPS Backend
template <int nbit>
void check_linear_mps_args(
    const Tensor& A,
    const Tensor& B,
    int64_t group_size,
    const Tensor& S,
    const Tensor& Z) {
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(
      A.dtype() == at::kBFloat16 || A.dtype() == at::kHalf ||
          A.dtype() == at::kFloat,
      __func__,
      " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(
      B.dtype() == at::kByte, __func__, " : expect B to be uint8 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(
      B.size(1) == (K / 8) * nbit,
      __func__,
      " : expect B.size(1) == ",
      (K / 8) * nbit);

  TORCH_CHECK(K % 8 == 0, __func__, ": expect K to be multiple of 8, got ", K);

  TORCH_CHECK(N % 4 == 0, __func__, ": expect N to be multiple of 4, got ", N);

  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128 ||
          group_size == 256,
      __func__,
      ": expect group_size to be 32, 64, 128 or 256, got ",
      group_size);

  TORCH_CHECK(
      S.dim() == 2 && S.size(1) == N,
      __func__,
      ": expect S to be 2d tensor with shape [:, ",
      N,
      "]");
  TORCH_CHECK(S.is_contiguous(), __func__, " : expect S to be contiguous.");

  TORCH_CHECK(
      Z.dim() == 2 && Z.size(1) == N,
      __func__,
      ": expect Z to be 2d tensor with shape [:, ",
      N,
      "]");
  TORCH_CHECK(Z.is_contiguous(), __func__, " : expect Z to be contiguous.");
}

template <int nbit>
Tensor linear_mps_kernel_out(
    const Tensor& A,
    const Tensor& B,
    int64_t group_size,
    const Tensor& S,
    const Tensor& Z,
    Tensor& C) {
  TORCH_CHECK(
      A.is_mps(), __func__, ": A is on ", A.device(), " but expected on mps");
  TORCH_CHECK(
      B.is_mps(), __func__, ": B is on ", B.device(), " but expected on mps");
  TORCH_CHECK(
      S.is_mps(), __func__, ": S is on ", S.device(), " but expected on mps");
  TORCH_CHECK(
      Z.is_mps(), __func__, ": Z is on ", Z.device(), " but expected on mps");
  TORCH_CHECK(
      C.is_mps(), __func__, ": Z is on ", Z.device(), " but expected on mps");

  check_linear_mps_args<nbit>(A, B, group_size, S, Z);

  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  LowBitQuantWeights<nbit>::linear(
      getMTLBufferStorage(A),
      getMTLBufferStorage(B),
      group_size,
      getMTLBufferStorage(S),
      getMTLBufferStorage(Z),
      getMTLBufferStorage(C),
      M,
      K,
      N,
      scalarToMetalTypeString(A));

  return C;
}

template <int nbit>
Tensor linear_mps_kernel(
    const Tensor& A,
    const Tensor& B,
    int64_t group_size,
    const Tensor& S,
    const Tensor& Z) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto C = at::empty({M, N}, A.options());
  return linear_mps_kernel_out<nbit>(A, B, group_size, S, Z, C);
}

template <int nbit>
Tensor linear_mps_kernel_meta(
    const Tensor& A,
    const Tensor& B,
    int64_t group_size,
    const Tensor& S,
    const Tensor& Z) {
  TORCH_CHECK(
      A.is_meta(), __func__, ": A is on ", A.device(), " but expected on meta");
  TORCH_CHECK(
      B.is_meta(), __func__, ": B is on ", B.device(), " but expected on meta");
  TORCH_CHECK(
      S.is_meta(), __func__, ": S is on ", S.device(), " but expected on meta");
  TORCH_CHECK(
      Z.is_meta(), __func__, ": Z is on ", Z.device(), " but expected on meta");

  check_linear_mps_args<nbit>(A, B, group_size, S, Z);

  auto M = A.size(0);
  auto N = B.size(0);

  return at::empty({M, N}, A.options()).to("meta");
}

// LowBit Packing on CPU Backend
template <int nbit>
Tensor pack_weights_cpu_kernel(const Tensor& W) {
  auto N = W.size(0);
  auto K = W.size(1);
  auto B = at::empty({N, nbit * K / 8}, W.options());

  uint8_t* w_ptr = W.data_ptr<uint8_t>();
  uint8_t* b_ptr = B.data_ptr<uint8_t>();

  LowBitQuantWeights<nbit>::pack(w_ptr, b_ptr, N, K);

  return B;
}

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.def("_pack_weight_1bit(Tensor W) -> Tensor");
  m.def("_pack_weight_2bit(Tensor W) -> Tensor");
  m.def("_pack_weight_3bit(Tensor W) -> Tensor");
  m.def("_pack_weight_4bit(Tensor W) -> Tensor");
  m.def("_pack_weight_5bit(Tensor W) -> Tensor");
  m.def("_pack_weight_6bit(Tensor W) -> Tensor");
  m.def("_pack_weight_7bit(Tensor W) -> Tensor");
  m.def(
      "_linear_fp_act_1bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_2bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_3bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_4bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_5bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_6bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_7bit_weight(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z) -> Tensor");
  m.def(
      "_linear_fp_act_1bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_linear_fp_act_2bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_linear_fp_act_3bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_linear_fp_act_4bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_linear_fp_act_5bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_linear_fp_act_6bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "_linear_fp_act_7bit_weight.out(Tensor A, Tensor B, int group_size, Tensor S, Tensor Z, *, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("_pack_weight_1bit", &pack_weights_cpu_kernel<1>);
  m.impl("_pack_weight_2bit", &pack_weights_cpu_kernel<2>);
  m.impl("_pack_weight_3bit", &pack_weights_cpu_kernel<3>);
  m.impl("_pack_weight_4bit", &pack_weights_cpu_kernel<4>);
  m.impl("_pack_weight_5bit", &pack_weights_cpu_kernel<5>);
  m.impl("_pack_weight_6bit", &pack_weights_cpu_kernel<6>);
  m.impl("_pack_weight_7bit", &pack_weights_cpu_kernel<7>);
}

TORCH_LIBRARY_IMPL(torchao, MPS, m) {
  m.impl("_linear_fp_act_1bit_weight", &linear_mps_kernel<1>);
  m.impl("_linear_fp_act_2bit_weight", &linear_mps_kernel<2>);
  m.impl("_linear_fp_act_3bit_weight", &linear_mps_kernel<3>);
  m.impl("_linear_fp_act_4bit_weight", &linear_mps_kernel<4>);
  m.impl("_linear_fp_act_5bit_weight", &linear_mps_kernel<5>);
  m.impl("_linear_fp_act_6bit_weight", &linear_mps_kernel<6>);
  m.impl("_linear_fp_act_7bit_weight", &linear_mps_kernel<7>);
  m.impl("_linear_fp_act_1bit_weight.out", &linear_mps_kernel_out<1>);
  m.impl("_linear_fp_act_2bit_weight.out", &linear_mps_kernel_out<2>);
  m.impl("_linear_fp_act_3bit_weight.out", &linear_mps_kernel_out<3>);
  m.impl("_linear_fp_act_4bit_weight.out", &linear_mps_kernel_out<4>);
  m.impl("_linear_fp_act_5bit_weight.out", &linear_mps_kernel_out<5>);
  m.impl("_linear_fp_act_6bit_weight.out", &linear_mps_kernel_out<6>);
  m.impl("_linear_fp_act_7bit_weight.out", &linear_mps_kernel_out<7>);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  m.impl("_linear_fp_act_1bit_weight", &linear_mps_kernel_meta<1>);
  m.impl("_linear_fp_act_2bit_weight", &linear_mps_kernel_meta<2>);
  m.impl("_linear_fp_act_3bit_weight", &linear_mps_kernel_meta<3>);
  m.impl("_linear_fp_act_4bit_weight", &linear_mps_kernel_meta<4>);
  m.impl("_linear_fp_act_5bit_weight", &linear_mps_kernel_meta<5>);
  m.impl("_linear_fp_act_6bit_weight", &linear_mps_kernel_meta<6>);
  m.impl("_linear_fp_act_7bit_weight", &linear_mps_kernel_meta<7>);
}

} // namespace torchao::kernels::mps::lowbit::aten
