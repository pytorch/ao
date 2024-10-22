// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/embedding_xbit/op_embedding_xbit-impl.h>

#define DEFINE_OP(weight_nbit)             \
  Tensor _op_out_##weight_nbit(            \
      RuntimeContext& ctx,                 \
      const Tensor& packed_weight_qvals,   \
      const Tensor& num_embeddings_tensor, \
      const Tensor& embedding_dim_tensor,  \
      const Tensor& weight_scales,         \
      const Tensor& weight_zeros,          \
      const Tensor& indices,               \
      Tensor& out) {                       \
    (void)ctx;                             \
    embedding_out_cpu<weight_nbit>(        \
        packed_weight_qvals,               \
        num_embeddings_tensor,             \
        embedding_dim_tensor,              \
        weight_scales,                     \
        weight_zeros,                      \
        indices,                           \
        out);                              \
    return out;                            \
  }                                        \
  EXECUTORCH_LIBRARY(                      \
      torchao, "_embedding_" #weight_nbit "bit.out", _op_out_##weight_nbit)

DEFINE_OP(1);
DEFINE_OP(2);
DEFINE_OP(3);
DEFINE_OP(4);
DEFINE_OP(5);
DEFINE_OP(6);
DEFINE_OP(7);
