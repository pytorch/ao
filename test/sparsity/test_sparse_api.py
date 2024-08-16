import copy
import logging
import unittest

import torch
from torch import nn

from torchao.sparsity import (
    apply_fake_sparsity,
    sparsify_,
    int8_dynamic_activation_int8_semi_sparse_weight,
    semi_sparse_weight,
)
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
    int8_dynamic_activation_int8_weight,
    quantize_,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_3
from torch.testing._internal.common_utils import TestCase


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

class TestSemiStructuredSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        dense_result = model(input)

        sparsify_(model, semi_sparse_weight())
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

class TestQuantSemiSparse(TestCase):

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quant_semi_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )
        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)
        quantize_(model_copy, int8_dynamic_activation_int8_weight())
        dense_result = model_copy(input)

        quantize_(model, int8_dynamic_activation_int8_semi_sparse_weight())
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    unittest.main()
