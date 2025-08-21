// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/embedding_lut/op_embedding_groupwise_lowbit_lut-impl.h>

#define DEFINE_LUT_OP(weight_nbit)              \
  Tensor _op_lut_out_##weight_nbit(             \
      RuntimeContext& ctx,                      \
      const Tensor& packed_weights,             \
      const Tensor& indices,                    \
      const int64_t& num_embeddings,            \
      const int64_t& embedding_dim,             \
      const int64_t& scale_group_size,          \
      const int64_t& lut_group_size,            \
      const bool& has_scales,                   \
      Tensor& out) {                            \
    (void)ctx;                                  \
    embedding_out_cpu<weight_nbit>(             \
        packed_weights,                         \
        indices,                                \
        num_embeddings,                         \
        embedding_dim,                          \
        scale_group_size,                       \
        lut_group_size,                         \
        has_scales,                             \
        out);                                   \
    return out;                                 \
  }                                             \
  EXECUTORCH_LIBRARY(                           \
      torchao,                                  \
      "_embedding_lut_" #weight_nbit "bit.out", \
      _op_lut_out_##weight_nbit)

DEFINE_LUT_OP(1);
DEFINE_LUT_OP(2);
DEFINE_LUT_OP(3);
DEFINE_LUT_OP(4);
DEFINE_LUT_OP(5);
DEFINE_LUT_OP(6);
DEFINE_LUT_OP(7);
DEFINE_LUT_OP(8);
