#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h>

#define DEFINE_OP(weight_nbit)                                                 \
  Tensor _op_out_0zp_##weight_nbit(                                            \
      RuntimeContext &ctx, const Tensor &activations,                          \
      const Tensor &packed_weights, const int64_t &group_size,                 \
      const int64_t &n, const int64_t &k, Tensor &out) {                       \
    (void)ctx;                                                                 \
    linear_out_cpu<weight_nbit, false>(activations, packed_weights,            \
                                       group_size, n, k, out);                 \
    return out;                                                                \
  }                                                                            \
  Tensor _op_out_zp_##weight_nbit(                                             \
      RuntimeContext &ctx, const Tensor &activations,                          \
      const Tensor &packed_weights, const int64_t &group_size,                 \
      const int64_t &n, const int64_t &k, Tensor &out) {                       \
    (void)ctx;                                                                 \
    linear_out_cpu<weight_nbit, true>(activations, packed_weights, group_size, \
                                      n, k, out);                              \
    return out;                                                                \
  }

#define REGISTER_0ZP(weight_nbit)                                              \
  EXECUTORCH_LIBRARY(torchao,                                                  \
                     "_linear_8bit_act_" #weight_nbit "bit0zp_weight.out",     \
                     _op_out_0zp_##weight_nbit)

#define REGISTER_ZP(weight_nbit)                                               \
  EXECUTORCH_LIBRARY(torchao,                                                  \
                     "_linear_8bit_act_" #weight_nbit "bit_weight.out",        \
                     _op_out_zp_##weight_nbit)

// This looks a bit ridiculous, but I could not get it to compile with two
// EXECUTORCH_LIBRARY nested inside DEFINE_OP
DEFINE_OP(1)
REGISTER_0ZP(1);
REGISTER_ZP(1);

DEFINE_OP(2)
REGISTER_0ZP(2);
REGISTER_ZP(2);

DEFINE_OP(3)
REGISTER_0ZP(3);
REGISTER_ZP(3);

DEFINE_OP(4)
REGISTER_0ZP(4);
REGISTER_ZP(4);

DEFINE_OP(5)
REGISTER_0ZP(5);
REGISTER_ZP(5);

DEFINE_OP(6)
REGISTER_0ZP(6);
REGISTER_ZP(6);

DEFINE_OP(7)
REGISTER_0ZP(7);
REGISTER_ZP(7);

DEFINE_OP(8)
REGISTER_0ZP(8);
REGISTER_ZP(8);
