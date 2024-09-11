#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("marlin_24_gemm(Tensor x, Tensor weight_marlin, Tensor meta, Tensor s, Tensor workspace, int bits, int size_m, int size_n, int size_k) -> Tensor");
}
