// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/embedding_xbit/op_embedding_xbit-impl.h>

#define DEFINE_OP(weight_nbit)                                                                                                                                            \
  m.def("_pack_embedding_" #weight_nbit "bit(Tensor weight_qvals) -> Tensor");                                                                                            \
  m.def(                                                                                                                                                                  \
      "_embedding_" #weight_nbit                                                                                                                                          \
      "bit(Tensor packed_weight_qvals, Tensor num_embeddings_tensor, Tensor embedding_dim_tensor, Tensor weight_scales, Tensor weight_zeros, Tensor indices) -> Tensor"); \
  m.def(                                                                                                                                                                  \
      "_embedding_" #weight_nbit                                                                                                                                          \
      "bit.out(Tensor packed_weight_qvals, Tensor num_embeddings_tensor, Tensor embedding_dim_tensor, Tensor weight_scales, Tensor weight_zeros, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)");

#define DEFINE_CPU_IMPL(weight_nbit)                                     \
  m.impl(                                                                \
      "_pack_embedding_" #weight_nbit "bit",                             \
      &pack_embedding_cpu<weight_nbit>);                                 \
  m.impl("_embedding_" #weight_nbit "bit", &embedding_cpu<weight_nbit>); \
  m.impl("_embedding_" #weight_nbit "bit.out", &embedding_out_cpu<weight_nbit>);

#define DEFINE_META_IMPL(weight_nbit)        \
  m.impl(                                    \
      "_pack_embedding_" #weight_nbit "bit", \
      &pack_embedding_meta<weight_nbit>);    \
  m.impl("_embedding_" #weight_nbit "bit", &embedding_meta<weight_nbit>);

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  DEFINE_OP(1);
  DEFINE_OP(2);
  DEFINE_OP(3);
  DEFINE_OP(4);
  DEFINE_OP(5);
  DEFINE_OP(6);
  DEFINE_OP(7);
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  DEFINE_CPU_IMPL(1);
  DEFINE_CPU_IMPL(2);
  DEFINE_CPU_IMPL(3);
  DEFINE_CPU_IMPL(4);
  DEFINE_CPU_IMPL(5);
  DEFINE_CPU_IMPL(6);
  DEFINE_CPU_IMPL(7);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  DEFINE_META_IMPL(1);
  DEFINE_META_IMPL(2);
  DEFINE_META_IMPL(3);
  DEFINE_META_IMPL(4);
  DEFINE_META_IMPL(5);
  DEFINE_META_IMPL(6);
  DEFINE_META_IMPL(7);
}
