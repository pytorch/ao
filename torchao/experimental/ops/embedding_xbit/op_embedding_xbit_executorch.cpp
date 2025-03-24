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
      const int64_t& num_embeddings,       \
      const int64_t& embedding_dim,        \
      const Tensor& weight_scales,         \
      const Tensor& weight_zeros,          \
      const Tensor& indices,               \
      Tensor& out) {                       \
    (void)ctx;                             \
    embedding_out_cpu<weight_nbit>(        \
        packed_weight_qvals,               \
        num_embeddings,                    \
        embedding_dim,                     \
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
DEFINE_OP(8);

#define DEFINE_SHARED_OP(weight_nbit)                    \
  Tensor _shared_op_out_##weight_nbit(                   \
      RuntimeContext& ctx,                               \
      const Tensor& packed_weights,                      \
      const int64_t& group_size,                         \
      const int64_t& n,                                  \
      const int64_t& k,                                  \
      const Tensor& indices,                             \
      Tensor& out) {                                     \
    (void)ctx;                                           \
    shared_embedding_out_cpu<weight_nbit>(               \
        packed_weights, group_size, n, k, indices, out); \
    return out;                                          \
  }                                                      \
  EXECUTORCH_LIBRARY(                                    \
      torchao,                                           \
      "_shared_embedding_" #weight_nbit "bit.out",       \
      _shared_op_out_##weight_nbit)

DEFINE_SHARED_OP(1);
DEFINE_SHARED_OP(2);
DEFINE_SHARED_OP(3);
DEFINE_SHARED_OP(4);
DEFINE_SHARED_OP(5);
DEFINE_SHARED_OP(6);
DEFINE_SHARED_OP(7);
DEFINE_SHARED_OP(8);
