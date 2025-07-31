#include <torchao/experimental/ops/groupwise_lowbit_weight_lut/op_groupwise_lowbit_weight_lut-impl.h>

#define DEFINE_OP(weight_nbit)                         \
  Tensor _op_out_##weight_nbit(                        \
      RuntimeContext& ctx,                             \
      const Tensor& activations,                       \
      const Tensor& packed_weights,                    \
      const int64_t& scale_group_size,                 \
      const int64_t& lut_group_size,                   \
      const int64_t& n,                                \
      const int64_t& k,                                \
      Tensor& out) {                                   \
    (void)ctx;                                         \
    linear_out_cpu<weight_nbit>(                       \
        activations,                                   \
        packed_weights,                                \
        scale_group_size,                              \
        lut_group_size,                                \
        n,                                             \
        k,                                             \
        out);                                          \
    return out;                                        \
  }                                                    \
  EXECUTORCH_LIBRARY(                                  \
      torchao,                                         \
      "_linear_groupwise_" #weight_nbit "bit_weight_with_lut.out", \
      _op_out_##weight_nbit)

DEFINE_OP(1);
DEFINE_OP(2);
DEFINE_OP(3);
DEFINE_OP(4);
