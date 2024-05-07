#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("fp16act_fp6weight_linear(Tensor _weights, Tensor _scales, int splitK = 1) -> Tensor");
  m.def("fp6_weight_prepacking_cpu(Tensor fp6_tensor) -> Tensor");
  m.def("fp6_weight_dequant_cpu(Tensor fp6_tensor, Tensor fp16_scale) -> Tensor");
}
