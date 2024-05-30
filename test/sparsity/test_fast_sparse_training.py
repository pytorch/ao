import logging
import unittest

import torch
from torch import nn
import copy
import torch.nn.functional as F

from torchao.sparsity import apply_fake_sparsity, apply_sparse_semi_structured
from torchao.sparsity.prototype.fast_sparse_training import swap_linear_with_semi_sparse_linear_
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_3
from torch.testing._internal.common_utils import TestCase
from torch.sparse import SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
torch.set_printoptions(precision=3, edgeitems=4, linewidth=1000000000)
class TestQuantSemiSparse(TestCase):

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

        sparse_config = {
            "linear1": True,
            "linear2": True,
        }

        swap_linear_with_semi_sparse_linear_(model_c, sparse_config)
        sparse_result = model_c(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)

        dense_result.backward(grad)
        sparse_result.backward(grad)

        # check grad
        assert torch.allclose(model.linear1.weight.grad, model_c.linear1.weight.grad, rtol=1e-1, atol=1e-1)
        assert torch.allclose(model.linear2.weight.grad, model_c.linear2.weight.grad, rtol=1e-1, atol=1e-1)

if __name__ == "__main__":
    unittest.main()
