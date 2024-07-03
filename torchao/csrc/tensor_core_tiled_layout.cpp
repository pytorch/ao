#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("unpack_tensor_core_tiled_layout(Tensor packed_w, int inner_k_tiles) -> Tensor");
  m.def("dequantize_tensor_core_tiled_layout(Tensor packed_w, Tensor scales_and_zeros, int group_size, int inner_k_tiles) -> Tensor");

}
