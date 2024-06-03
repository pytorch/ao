import logging
import unittest
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_utils import TestCase

from torchao.sparsity.prototype.training import (
    swap_linear_with_semi_sparse_linear_,
    swap_semi_sparse_linear_with_linear_,
    SemiSparseLinear
)
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3
if TORCH_VERSION_AFTER_2_3:
    from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256, bias=False)
        self.linear2 = nn.Linear(256, 128, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x

class TestRuntimeSemiStructuredSparsity(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_runtime_weight_sparsification(self):
        input = torch.rand((128, 128)).half().cuda()
        grad = torch.rand((128, 128)).half().cuda()
        model = TestModel().half().cuda()
        model_c = copy.deepcopy(model)

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                sparse = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(mod.weight.detach()).to_dense()
                mod.weight = nn.Parameter(sparse)
        
        dense_result = model(input)

        # map from fqn to replacement linear module
        sparse_config = {
            "linear1": SemiSparseLinear,
            "linear2": SemiSparseLinear,
        }

        swap_linear_with_semi_sparse_linear_(model_c, sparse_config)
        sparse_result = model_c(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)

        dense_result.backward(grad)
        sparse_result.backward(grad)

        # check grad
        assert torch.allclose(model.linear1.weight.grad, model_c.linear1.weight.grad, rtol=1e-1, atol=1e-1)
        assert torch.allclose(model.linear2.weight.grad, model_c.linear2.weight.grad, rtol=1e-1, atol=1e-1)

        # check that swap back works
        swap_semi_sparse_linear_with_linear_(model_c)
        for name, mod in model_c.named_modules():
            assert not isinstance(mod, SemiSparseLinear)


if __name__ == "__main__":
    unittest.main()
