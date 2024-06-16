#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("int4mv(Tensor A, Tensor B, int groupSize, Tensor scalesAndZeros) -> Tensor");
  m.def("int4mv.out(Tensor A, Tensor B, int groupSize, Tensor scalesAndZeros, *, Tensor(a!) out) -> Tensor(a!)");
}
