// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h>

#define DEFINE_OP(weight_nbit)                                                                                                                 \
  m.def(                                                                                                                                       \
      "_pack_8bit_act_" #weight_nbit                                                                                                   \
      "bit0zp_weight(Tensor weight_qvals, Tensor weight_scales, Tensor group_size) -> Tensor");                                                \
  m.def(                                                                                                                                       \
      "_pack_8bit_act_" #weight_nbit                                                                                                   \
      "bit_weight(Tensor weight_qvals, Tensor weight_scales, Tensor weight_zeros, Tensor group_size) -> Tensor");                              \
  m.def(                                                                                                                                       \
      "_linear_8bit_act_" #weight_nbit                                                                                                         \
      "bit0zp_weight(Tensor activations, Tensor packed_weights, Tensor group_size, Tensor n, Tensor k) -> Tensor");                            \
  m.def(                                                                                                                                       \
      "_linear_8bit_act_" #weight_nbit                                                                                                         \
      "bit_weight(Tensor activations, Tensor packed_weights, Tensor group_size, Tensor n, Tensor k) -> Tensor");                               \
  m.def(                                                                                                                                       \
      "_linear_8bit_act_" #weight_nbit                                                                                                         \
      "bit0zp_weight.out(Tensor activations, Tensor packed_weights, Tensor group_size, Tensor n, Tensor k, *, Tensor(a!) out) -> Tensor(a!)"); \
  m.def(                                                                                                                                       \
      "_linear_8bit_act_" #weight_nbit                                                                                                         \
      "bit_weight.out(Tensor activations, Tensor packed_weights, Tensor group_size, Tensor n, Tensor k, *, Tensor(a!) out) -> Tensor(a!)")

#define DEFINE_CPU_IMPL(weight_nbit)                          \
  m.impl(                                                     \
      "_pack_8bit_act_" #weight_nbit "bit0zp_weight", \
      &pack_weights_without_zeros_cpu<weight_nbit>);          \
  m.impl(                                                     \
      "_pack_8bit_act_" #weight_nbit "bit_weight",    \
      &pack_weights_with_zeros_cpu<weight_nbit>);             \
  m.impl(                                                     \
      "_linear_8bit_act_" #weight_nbit "bit0zp_weight",       \
      &linear_cpu<weight_nbit, false>);                       \
  m.impl(                                                     \
      "_linear_8bit_act_" #weight_nbit "bit_weight",          \
      &linear_cpu<weight_nbit, true>);                        \
  m.impl(                                                     \
      "_linear_8bit_act_" #weight_nbit "bit0zp_weight.out",   \
      &linear_out_cpu<weight_nbit, false>);                   \
  m.impl(                                                     \
      "_linear_8bit_act_" #weight_nbit "bit_weight.out",      \
      &linear_out_cpu<weight_nbit, true>)

#define DEFINE_META_IMPL(weight_nbit)                         \
  m.impl(                                                     \
      "_pack_8bit_act_" #weight_nbit "bit0zp_weight", \
      &pack_weights_without_zeros_meta<weight_nbit>);         \
  m.impl(                                                     \
      "_pack_8bit_act_" #weight_nbit "bit_weight",    \
      &pack_weights_with_zeros_meta<weight_nbit>);            \
  m.impl(                                                     \
      "_linear_8bit_act_" #weight_nbit "bit0zp_weight",       \
      &linear_meta<weight_nbit, false>);                      \
  m.impl(                                                     \
      "_linear_8bit_act_" #weight_nbit "bit_weight",          \
      &linear_meta<weight_nbit, true>);

TORCH_LIBRARY(torchao, m) {
  DEFINE_OP(1);
  DEFINE_OP(2);
  DEFINE_OP(3);
  DEFINE_OP(4);
  DEFINE_OP(5);
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  DEFINE_CPU_IMPL(1);
  DEFINE_CPU_IMPL(2);
  DEFINE_CPU_IMPL(3);
  DEFINE_CPU_IMPL(4);
  DEFINE_CPU_IMPL(5);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  DEFINE_META_IMPL(1);
  DEFINE_META_IMPL(2);
  DEFINE_META_IMPL(3);
  DEFINE_META_IMPL(4);
  DEFINE_META_IMPL(5);
}
