#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("unpack_int4_to_int(Tensor packed_w, int innerKTiles) -> Tensor");
  m.def("dequantize_int4(Tensor packed_w, Tensor scales_and_zeros, int group_size, int innerKTiles) -> Tensor");

}
