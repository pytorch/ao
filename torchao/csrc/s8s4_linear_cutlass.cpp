#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("s8s4_linear_cutlass(Tensor input, Tensor input_scale, Tensor weight, Tensor weight_scale, Tensor bias) -> Tensor");
}
