#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  m.impl_abstract_pystub("torchao.ops");
  m.def(
      "paged_attention(Tensor (a!)out, Tensor (a!)query, Tensor (a!)key_cache, Tensor (a!)value_cache,\
       Tensor(a!) head_mapping, float scale, Tensor(a!) block_tables, Tensor(a!) context_lens, int block_size,\
       Tensor? attn_mask)-> ()");
}