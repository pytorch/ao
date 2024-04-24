import logging
import unittest
import copy

import torch
from torch import nn

from torchao.sparsity import apply_fake_sparsity, apply_sparse_semi_structured
from torchao.sparsity.prototype.dynamic_quant_sparse import Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight
from torchao.quantization.quant_api import (
    Int8DynamicallyQuantizedLinearWeight,
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
)
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from torch.testing._internal.common_utils import TestCase
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

class TestSemiStructuredSparse(TestCase):

    def test_sparse(self):
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        input = torch.rand((128, 128), device="cuda").half()
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

        apply_sparse_semi_structured(model)
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)


class TestQuantSemiSparse(TestCase):

    def test_quant_semi_sparse_proto(self):
        input = torch.rand((128, 128), device="cuda").half()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                # nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        dense_result = model(input)

        _replace_with_custom_fn_if_matches_filter(model, _get_subclass_inserter(Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight), _is_linear)
        model = torch.compile(model, mode='max-autotune')
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)

    def test_quant_semi_sparse_composed(self):
        input = torch.rand((128, 128), device="cuda").half()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )
        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        dense_result = model(input)

        change_linear_weights_to_int8_dqtensors(model)
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and isinstance(mod.weight.data, Int8DynamicallyQuantizedLinearWeight):
                mod.weight.data.int_data = to_sparse_semi_structured(mod.weight.data.int_data.t())
        sparse_result_composed = model(input)
        model = torch.compile(model, mode='max-autotune')

        assert torch.allclose(dense_result, sparse_result_composed, rtol=1e-1, atol=1e-1)

        _replace_with_custom_fn_if_matches_filter(model_copy, _get_subclass_inserter(Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight), _is_linear)
        model_copy = torch.compile(model_copy, mode='max-autotune')
        sparse_result_manual = model_copy(input)
        assert torch.allclose(sparse_result_composed, sparse_result_manual, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
