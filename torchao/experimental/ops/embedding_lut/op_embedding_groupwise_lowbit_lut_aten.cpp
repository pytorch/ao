// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/embedding_lut/op_embedding_groupwise_lowbit_lut-impl.h>

// This macro defines the operator signatures.
// The signatures now correctly match the C++ implementation.
#define DEFINE_LUT_OP(weight_nbit)                                          \
  m.def(                                                                    \
      "_pack_embedding_lut_" #weight_nbit                                   \
      "bit(Tensor weight_qval_idxs, Tensor luts, int scale_group_size, "    \
      "int lut_group_size, Tensor? weight_scales) -> Tensor");              \
  m.def(                                                                    \
      "_embedding_lut_" #weight_nbit                                        \
      "bit(Tensor packed_weights, Tensor indices, int num_embeddings, "     \
      "int embedding_dim, int scale_group_size, int lut_group_size, "       \
      "bool has_scales) -> Tensor");                                        \
  m.def(                                                                    \
      "_embedding_lut_" #weight_nbit                                        \
      "bit.out(Tensor packed_weights, Tensor indices, int num_embeddings, " \
      "int embedding_dim, int scale_group_size, int lut_group_size, "       \
      "bool has_scales, *, Tensor(a!) out) -> Tensor(a!)");

// This macro registers the CPU implementations for the LUT-based operators.
#define DEFINE_CPU_IMPL(weight_nbit)                                   \
  m.impl(                                                              \
      "_pack_embedding_lut_" #weight_nbit "bit",                       \
      torch::dispatch(                                                 \
          c10::DispatchKey::CPU, &pack_embedding_cpu<weight_nbit>));    \
  m.impl(                                                              \
      "_embedding_lut_" #weight_nbit "bit",                            \
      torch::dispatch(                                                 \
          c10::DispatchKey::CPU, &embedding_cpu<weight_nbit>));         \
  m.impl(                                                              \
      "_embedding_lut_" #weight_nbit "bit.out",                        \
      torch::dispatch(                                                 \
          c10::DispatchKey::CPU, &embedding_out_cpu<weight_nbit>));

// This macro registers the Meta (device-agnostic) implementation for packing.
#define DEFINE_META_IMPL(weight_nbit)                                  \
  m.impl(                                                              \
      "_pack_embedding_lut_" #weight_nbit "bit",                       \
      torch::dispatch(                                                 \
          c10::DispatchKey::Meta, &pack_embedding_meta<weight_nbit>));

// Operator definitions
TORCH_LIBRARY_FRAGMENT(torchao, m) {
  DEFINE_LUT_OP(1);
  DEFINE_LUT_OP(2);
  DEFINE_LUT_OP(3);
  DEFINE_LUT_OP(4);
}

// CPU implementations
TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  DEFINE_CPU_IMPL(1);
  DEFINE_CPU_IMPL(2);
  DEFINE_CPU_IMPL(3);
  DEFINE_CPU_IMPL(4);
}

// Meta implementations
TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  DEFINE_META_IMPL(1);
  DEFINE_META_IMPL(2);
  DEFINE_META_IMPL(3);
  DEFINE_META_IMPL(4);
}
