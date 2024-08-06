#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def("marlin_24_mm(Tensor x, Tensor weight_marlin, Tensor meta, Tensor out, Tensor s, int prob_m, int prob_n, int prob_k, Tensor workspace, int group_size, int dev, int cuda_stream, int thread_k, int thread_m, int sms, int max_par) -> int");
}
