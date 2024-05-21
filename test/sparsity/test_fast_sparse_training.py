import logging
import unittest

import torch
from torch import nn
import copy

from torchao.sparsity import apply_fake_sparsity, apply_sparse_semi_structured
from torchao.sparsity.prototype.fast_sparse_training import swap_linear_with_semi_sparse_linear_
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
)
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3
from torch.testing._internal.common_utils import TestCase

from torch.sparse._semi_structured_conversions import (
   _sparse_semi_structured_tile
)
from torch.sparse import SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
torch.set_printoptions(precision=3, edgeitems=4, linewidth=1000000000)
class TestQuantSemiSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quant_semi_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        grad = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )
        model_c = copy.deepcopy(model)

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                sparse = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(mod.weight.detach()).to_dense()
                mod.weight = nn.Parameter(sparse)
                print(name)
                print(mod.weight)
                break
        dense_result = model(input)


        swap_linear_with_semi_sparse_linear_(model_c, {".0"})
        sparse_result = model_c(input)


        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)

        dense_result.backward(grad)
        sparse_result.backward(grad)

        one = model[0].weight.grad
        two = model_c[0].weight.grad
        assert torch.allclose(one, two, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
