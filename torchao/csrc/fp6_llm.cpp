#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("fp6_llm_linear(Tensor _in_feats, Tensor _weights, Tensor _scales, int splitK) -> Tensor");
}
