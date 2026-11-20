// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h>
#include <torchao/csrc/cpu/torch_free_kernels/weight_packing/weight_packing.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace {

using UnpackedWeights =
    std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>;

UnpackedWeights unpack_intx_weights_cpu(
    const Tensor& packed_weights,
    int64_t group_size,
    int64_t n,
    int64_t k) {
  using torchao::ops::PackedWeightsHeader;
  using torchao::ops::PackedWeightsType;
  using namespace torchao::ops::linear_8bit_act_xbit_weight;
  TORCH_CHECK(
      packed_weights.is_cpu() && packed_weights.scalar_type() == at::kChar &&
          packed_weights.dim() == 1 && packed_weights.is_contiguous(),
      "unpack expects a contiguous one-dimensional CPU int8 packed buffer");
  TORCH_CHECK(
      n > 0 && n <= std::numeric_limits<int>::max() - 7 && k > 0 &&
          k <= std::numeric_limits<int>::max() / 2 && group_size > 0 &&
          group_size % 16 == 0 && k % group_size == 0,
      "unpack expects positive n/k and a positive group divisible by 16 that divides k");
  TORCH_CHECK(packed_weights.numel() >= PackedWeightsHeader::size(), "truncated packed header");
  const auto* data = packed_weights.data_ptr<int8_t>();
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(data) % alignof(int) == 0 &&
          *reinterpret_cast<const int*>(data) == PackedWeightsHeader::magic,
      "unaligned or invalid packed header");
  const auto header = PackedWeightsHeader::read(data);
  const auto format = PackedWeightsFormat::from_packed_weights_header(header);
  check_format(format, PackedWeightsType::linear_8bit_act_xbit_weight_universal, format.weight_nbit);
  TORCH_CHECK(
      format.weight_nbit >= 1 && format.weight_nbit <= 8 && format.nr == 8 &&
          format.kr == 16 && format.sr == 2 && header == format.to_packed_weights_header(),
      "unpack supports only canonical universal INT1-INT8 nr=8/kr=16/sr=2 headers");
  const int n_int = static_cast<int>(n);
  const int k_int = static_cast<int>(k);
  const int group_size_int = static_cast<int>(group_size);
  const auto payload_size = torchao::weight_packing::packed_weights_size(
      n_int, k_int, group_size_int, format.weight_nbit, format.has_weight_zeros,
      format.has_bias, format.nr, format.kr, format.sr);
  TORCH_CHECK(
      static_cast<size_t>(packed_weights.numel()) == PackedWeightsHeader::size() + payload_size,
      "packed buffer size does not match n/k/group and header");
  const auto options = packed_weights.options();
  UnpackedWeights outputs{
      at::empty({n, k}, options),
      at::empty({n, k / group_size}, options.dtype(at::kFloat)),
      format.has_weight_zeros
          ? std::optional<Tensor>(at::empty({n, k / group_size}, options))
          : std::nullopt,
      format.has_bias
          ? std::optional<Tensor>(at::empty({n}, options.dtype(at::kFloat)))
          : std::nullopt};
  const auto* payload = data + PackedWeightsHeader::size();
  auto& [qvals, scales, zeros, bias] = outputs;
  auto* qvals_ptr = qvals.data_ptr<int8_t>();
  auto* scales_ptr = scales.data_ptr<float>();
  auto* zeros_ptr = zeros ? zeros->data_ptr<int8_t>() : nullptr;
  auto* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;
#define UNPACK_INTX_CASE(bits) \
  case bits: \
    torchao::weight_packing::unpack_weights<bits, 8, 16, 2>( \
        qvals_ptr, scales_ptr, zeros_ptr, bias_ptr, n_int, k_int, group_size_int, \
        format.has_weight_zeros, format.has_bias, payload); \
    break
  switch (format.weight_nbit) {
    UNPACK_INTX_CASE(1);
    UNPACK_INTX_CASE(2);
    UNPACK_INTX_CASE(3);
    UNPACK_INTX_CASE(4);
    UNPACK_INTX_CASE(5);
    UNPACK_INTX_CASE(6);
    UNPACK_INTX_CASE(7);
    UNPACK_INTX_CASE(8);
    default:
      TORCH_CHECK(false, "Unsupported universal weight bit width");
  }
#undef UNPACK_INTX_CASE
  return outputs;
}

} // namespace

#define DEFINE_OP(weight_nbit)                                                                                                             \
  m.def(                                                                                                                                   \
      "_pack_8bit_act_" #weight_nbit                                                                                                       \
      "bit_weight(Tensor weight_qvals, Tensor weight_scales, Tensor? weight_zeros, int group_size, Tensor? bias, str? target) -> Tensor"); \
  m.def(                                                                                                                                   \
      "_linear_8bit_act_" #weight_nbit                                                                                                     \
      "bit_weight(Tensor activations, Tensor packed_weights, int group_size, int n, int k) -> Tensor");                                    \
  m.def(                                                                                                                                   \
      "_linear_8bit_act_" #weight_nbit                                                                                                     \
      "bit_weight.out(Tensor activations, Tensor packed_weights, int group_size, int n, int k, *, Tensor(a!) out) -> Tensor(a!)")

#define DEFINE_CPU_IMPL(weight_nbit)                     \
  m.impl(                                                \
      "_pack_8bit_act_" #weight_nbit "bit_weight",       \
      &pack_weights_cpu<weight_nbit>);                   \
  m.impl(                                                \
      "_linear_8bit_act_" #weight_nbit "bit_weight",     \
      &linear_cpu<weight_nbit>);                         \
  m.impl(                                                \
      "_linear_8bit_act_" #weight_nbit "bit_weight.out", \
      &linear_out_cpu<weight_nbit>)

#define DEFINE_META_IMPL(weight_nbit)                 \
  m.impl(                                             \
      "_pack_8bit_act_" #weight_nbit "bit_weight",    \
      &pack_weights_meta<weight_nbit>)

#define DEFINE_LUT_PACK_OP(weight_nbit)                                                                                                       \
  m.def(                                                                                                                                      \
      "_pack_8bit_act_" #weight_nbit                                                                                                          \
      "bit_weight_with_lut(Tensor weight_qval_ids, Tensor luts, Tensor weight_scales, int group_size, Tensor? bias, str? target) -> Tensor")

#define DEFINE_LUT_PACK_CPU_IMPL(weight_nbit)               \
  m.impl(                                                   \
      "_pack_8bit_act_" #weight_nbit "bit_weight_with_lut", \
      &pack_weights_with_lut_cpu<weight_nbit>)

#define DEFINE_LUT_PACK_META_IMPL(weight_nbit)              \
  m.impl(                                                   \
      "_pack_8bit_act_" #weight_nbit "bit_weight_with_lut", \
      &pack_weights_with_lut_meta<weight_nbit>)


TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.def("_unpack_intx_weights(Tensor packed_weights, int group_size, int n, int k) -> (Tensor, Tensor, Tensor?, Tensor?)");
  DEFINE_OP(1);
  DEFINE_OP(2);
  DEFINE_OP(3);
  DEFINE_OP(4);
  DEFINE_OP(5);
  DEFINE_OP(6);
  DEFINE_OP(7);
  DEFINE_OP(8);

  DEFINE_LUT_PACK_OP(1);
  DEFINE_LUT_PACK_OP(2);
  DEFINE_LUT_PACK_OP(3);
  DEFINE_LUT_PACK_OP(4);
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("_unpack_intx_weights", &unpack_intx_weights_cpu);
  DEFINE_CPU_IMPL(1);
  DEFINE_CPU_IMPL(2);
  DEFINE_CPU_IMPL(3);
  DEFINE_CPU_IMPL(4);
  DEFINE_CPU_IMPL(5);
  DEFINE_CPU_IMPL(6);
  DEFINE_CPU_IMPL(7);
  DEFINE_CPU_IMPL(8);

  DEFINE_LUT_PACK_CPU_IMPL(1);
  DEFINE_LUT_PACK_CPU_IMPL(2);
  DEFINE_LUT_PACK_CPU_IMPL(3);
  DEFINE_LUT_PACK_CPU_IMPL(4);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  DEFINE_META_IMPL(1);
  DEFINE_META_IMPL(2);
  DEFINE_META_IMPL(3);
  DEFINE_META_IMPL(4);
  DEFINE_META_IMPL(5);
  DEFINE_META_IMPL(6);
  DEFINE_META_IMPL(7);
  DEFINE_META_IMPL(8);

  DEFINE_LUT_PACK_META_IMPL(1);
  DEFINE_LUT_PACK_META_IMPL(2);
  DEFINE_LUT_PACK_META_IMPL(3);
  DEFINE_LUT_PACK_META_IMPL(4);
}
