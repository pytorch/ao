# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase

from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
    swap_semi_sparse_linear_with_linear,
)
from torchao.testing.model_architectures import ToyTwoLinearModel
from torchao.utils import is_fbcode


class TestRuntimeSemiStructuredSparsity(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    @unittest.skip("Temporarily skipping to unpin nightlies")
    def test_runtime_weight_sparsification(self):
        # need this import inside to not break 2.2 tests
        from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

        input = torch.rand((128, 128)).half().cuda()
        grad = torch.rand((128, 128)).half().cuda()
        model = ToyTwoLinearModel(128, 256, 128).half().cuda()
        model_c = copy.deepcopy(model)

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                sparse = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(
                    mod.weight.detach()
                ).to_dense()
                mod.weight = nn.Parameter(sparse)

        dense_result = model(input)

        # map from fqn to replacement linear module
        sparse_config = {
            "linear1": SemiSparseLinear,
            "linear2": SemiSparseLinear,
        }

        swap_linear_with_semi_sparse_linear(model_c, sparse_config)
        sparse_result = model_c(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)

        dense_result.backward(grad)
        sparse_result.backward(grad)

        # check grad
        assert torch.allclose(
            model.linear1.weight.grad, model_c.linear1.weight.grad, rtol=1e-1, atol=1e-1
        )
        assert torch.allclose(
            model.linear2.weight.grad, model_c.linear2.weight.grad, rtol=1e-1, atol=1e-1
        )

        # check that swap back works
        swap_semi_sparse_linear_with_linear(model_c)
        for name, mod in model_c.named_modules():
            assert not isinstance(mod, SemiSparseLinear)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_runtime_weight_sparsification_compile(self):
        # need this import inside to not break 2.2 tests
        from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

        input = torch.rand((128, 128)).half().cuda()
        grad = torch.rand((128, 128)).half().cuda()
        model = ToyTwoLinearModel(128, 256, 128).half().cuda()
        model_c = copy.deepcopy(model)

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                sparse = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(
                    mod.weight.detach()
                ).to_dense()
                mod.weight = nn.Parameter(sparse)

        model = torch.compile(model, fullgraph=True)
        dense_result = model(input)

        # map from fqn to replacement linear module
        sparse_config = {
            "linear1": SemiSparseLinear,
            "linear2": SemiSparseLinear,
        }

        swap_linear_with_semi_sparse_linear(model_c, sparse_config)
        model_c = torch.compile(model_c, fullgraph=True)
        sparse_result = model_c(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-1, atol=1e-1)

        dense_result.backward(grad)
        sparse_result.backward(grad)

        # check grad
        assert torch.allclose(
            model.linear1.weight.grad, model_c.linear1.weight.grad, rtol=1e-1, atol=1e-1
        )
        assert torch.allclose(
            model.linear2.weight.grad, model_c.linear2.weight.grad, rtol=1e-1, atol=1e-1
        )

        # check that swap back works
        swap_semi_sparse_linear_with_linear(model_c)
        for name, mod in model_c.named_modules():
            assert not isinstance(mod, SemiSparseLinear)


if __name__ == "__main__":
    unittest.main()
