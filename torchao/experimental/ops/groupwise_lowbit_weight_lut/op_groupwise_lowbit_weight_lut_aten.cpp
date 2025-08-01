// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/groupwise_lowbit_weight_lut/op_groupwise_lowbit_weight_lut-impl.h>

#define DEFINE_PACK_OP(weight_nbit)                                                                                                       \
  m.def(                                                                                                                                      \
      "_pack_groupwise_" #weight_nbit                                                                                                           \
      "bit_weight_with_lut(Tensor weight_qval_idxs, Tensor luts, int scale_group_size, int lut_group_size, Tensor? weight_scales, Tensor? bias, str? target) -> Tensor");

#define DEFINE_LINEAR_OP(weight_nbit)                                                                                                       \
  m.def(                                                                                                                                   \
      "_linear_groupwise_" #weight_nbit                                                                                                     \
      "bit_weight_with_lut(Tensor activations, Tensor packed_weights, int scale_group_size, int lut_group_size, int n, int k) -> Tensor");                                    \
  m.def(                                                                                                                                   \
      "_linear_groupwise_" #weight_nbit                                                                                                     \
      "bit_weight_with_lut.out(Tensor activations, Tensor packed_weights, int scale_group_size, int lut_group_size, int n, int k, *, Tensor(a!) out) -> Tensor(a!)");

#define DEFINE_PACK_CPU_IMPL(weight_nbit)               \
  m.impl(                                                   \
      "_pack_groupwise_" #weight_nbit "bit_weight_with_lut", \
      &pack_weights_with_lut_cpu<weight_nbit>);

#define DEFINE_PACK_META_IMPL(weight_nbit)              \
  m.impl(                                                   \
      "_pack_groupwise_" #weight_nbit "bit_weight_with_lut", \
      &pack_weights_with_lut_meta<weight_nbit>);

#define DEFINE_LINEAR_CPU_IMPL(weight_nbit)                     \
  m.impl(                                                \
      "_linear_groupwise_" #weight_nbit "bit_weight_with_lut",     \
      &linear_cpu<weight_nbit>);                         \
  m.impl(                                                \
      "_linear_groupwise_" #weight_nbit "bit_weight_with_lut.out", \
      &linear_out_cpu<weight_nbit>);

#define DEFINE_LINEAR_META_IMPL(weight_nbit)                     \
  m.impl(                                                \
      "_linear_groupwise_" #weight_nbit "bit_weight_with_lut",     \
      &linear_meta<weight_nbit>);                         \


TORCH_LIBRARY_FRAGMENT(torchao, m) {
  DEFINE_PACK_OP(1);
  DEFINE_PACK_OP(2);
  DEFINE_PACK_OP(3);
  DEFINE_PACK_OP(4);

  DEFINE_LINEAR_OP(1);
  DEFINE_LINEAR_OP(2);
  DEFINE_LINEAR_OP(3);
  DEFINE_LINEAR_OP(4);
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  DEFINE_PACK_CPU_IMPL(1);
  DEFINE_PACK_CPU_IMPL(2);
  DEFINE_PACK_CPU_IMPL(3);
  DEFINE_PACK_CPU_IMPL(4);

  DEFINE_LINEAR_CPU_IMPL(1);
  DEFINE_LINEAR_CPU_IMPL(2);
  DEFINE_LINEAR_CPU_IMPL(3);
  DEFINE_LINEAR_CPU_IMPL(4);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  DEFINE_PACK_META_IMPL(1);
  DEFINE_PACK_META_IMPL(2);
  DEFINE_PACK_META_IMPL(3);
  DEFINE_PACK_META_IMPL(4);

  DEFINE_LINEAR_META_IMPL(1);
  DEFINE_LINEAR_META_IMPL(2);
  DEFINE_LINEAR_META_IMPL(3);
  DEFINE_LINEAR_META_IMPL(4);
}
