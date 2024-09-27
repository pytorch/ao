// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/linear/linear_a8wxdq_op/linear_a8wxdq-impl.h>

#define DEFINE_OP(weight_nbit)                                                                                                     \
  m.def(                                                                                                                           \
      "_pack_weights_a8sz_w" #weight_nbit                                                                                          \
      "s(Tensor weight_qvals, Tensor weight_scales, Tensor group_size) -> Tensor");                                                \
  m.def(                                                                                                                           \
      "_pack_weights_a8sz_w" #weight_nbit                                                                                          \
      "sz(Tensor weight_qvals, Tensor weight_scales, Tensor weight_zeros, Tensor group_size) -> Tensor");                          \
  m.def(                                                                                                                           \
      "_linear_a8sz_w" #weight_nbit                                                                                                \
      "s(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");                            \
  m.def(                                                                                                                           \
      "_linear_a8sz_w" #weight_nbit                                                                                                \
      "sz(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");                           \
  m.def(                                                                                                                           \
      "_linear_a8sz_w" #weight_nbit                                                                                                \
      "s.out(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations, *, Tensor(a!) out) -> Tensor(a!)"); \
  m.def(                                                                                                                           \
      "_linear_a8sz_w" #weight_nbit                                                                                                \
      "sz.out(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations, *, Tensor(a!) out) -> Tensor(a!)")

#define DEFINE_CPU_IMPL(weight_nbit)                                          \
  m.impl(                                                                     \
      "_pack_weights_a8sz_w" #weight_nbit "s",                                \
      &pack_weights_without_zeros_cpu<weight_nbit>);                          \
  m.impl(                                                                     \
      "_pack_weights_a8sz_w" #weight_nbit "sz",                               \
      &pack_weights_with_zeros_cpu<weight_nbit>);                             \
  m.impl("_linear_a8sz_w" #weight_nbit "s", &linear_cpu<weight_nbit, false>); \
  m.impl("_linear_a8sz_w" #weight_nbit "sz", &linear_cpu<weight_nbit, true>); \
  m.impl(                                                                     \
      "_linear_a8sz_w" #weight_nbit "s.out",                                  \
      &linear_out_cpu<weight_nbit, false>);                                   \
  m.impl(                                                                     \
      "_linear_a8sz_w" #weight_nbit "sz.out",                                 \
      &linear_out_cpu<weight_nbit, true>)

#define DEFINE_META_IMPL(weight_nbit)                                          \
  m.impl(                                                                      \
      "_pack_weights_a8sz_w" #weight_nbit "s",                                 \
      &pack_weights_without_zeros_meta<weight_nbit>);                          \
  m.impl(                                                                      \
      "_pack_weights_a8sz_w" #weight_nbit "sz",                                \
      &pack_weights_with_zeros_meta<weight_nbit>);                             \
  m.impl("_linear_a8sz_w" #weight_nbit "s", &linear_meta<weight_nbit, false>); \
  m.impl("_linear_a8sz_w" #weight_nbit "sz", &linear_meta<weight_nbit, true>);

TORCH_LIBRARY(torchao, m) {
  DEFINE_OP(2);
  DEFINE_OP(3);
  DEFINE_OP(4);
  DEFINE_OP(5);
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  DEFINE_CPU_IMPL(2);
  DEFINE_CPU_IMPL(3);
  DEFINE_CPU_IMPL(4);
  DEFINE_CPU_IMPL(5);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  DEFINE_META_IMPL(2);
  DEFINE_META_IMPL(3);
  DEFINE_META_IMPL(4);
  DEFINE_META_IMPL(5);
}
