#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchao {

torch::Tensor toy_op2_cpu(
    torch::Tensor   _in_feats)
{
    std::cout<<"---- run into cpu 2 ----"<<std::endl;
    return _in_feats;

}


TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::toy_op2", &toy_op2_cpu);
}

} // namespace torchao
