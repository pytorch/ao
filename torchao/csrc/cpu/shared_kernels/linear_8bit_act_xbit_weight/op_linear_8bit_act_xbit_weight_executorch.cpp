#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h>

#define DEFINE_OP(weight_nbit)                               \
  Tensor _op_out_##weight_nbit(                              \
      RuntimeContext& ctx,                                   \
      const Tensor& activations,                             \
      const Tensor& packed_weights,                          \
      const int64_t& group_size,                             \
      const int64_t& n,                                      \
      const int64_t& k,                                      \
      Tensor& out) {                                         \
    (void)ctx;                                               \
    linear_out_cpu<weight_nbit>(                             \
        activations, packed_weights, group_size, n, k, out); \
    return out;                                              \
  }                                                          \
  EXECUTORCH_LIBRARY(                                        \
      torchao,                                               \
      "_linear_8bit_act_" #weight_nbit "bit_weight.out",     \
      _op_out_##weight_nbit)

DEFINE_OP(1);
DEFINE_OP(2);
DEFINE_OP(3);
DEFINE_OP(4);
DEFINE_OP(5);
DEFINE_OP(6);
DEFINE_OP(7);
DEFINE_OP(8);
