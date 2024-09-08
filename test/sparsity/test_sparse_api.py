import copy
import logging
import unittest

import torch
from torch import nn

from torchao.sparsity import (
    apply_fake_sparsity,
    sparsify_,
    semi_sparse_weight,
)
from torchao.dtypes import MarlinSparseLayoutType, SemiSparseLayoutType
from torchao.quantization.quant_api import (
    int8_dynamic_activation_int8_weight,
    quantize_,
    int4_weight_only,
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

        quantize_(model, int8_dynamic_activation_int8_weight(layout_type=SemiSparseLayoutType()))
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_sparse_marlin(self):
        input = torch.rand((256, 256)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(256, 1024),
                nn.Linear(1024, 256),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, atol=3e-1), "Results are not close"

if __name__ == "__main__":
    unittest.main()
