# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfRocmVersionLessThan

from torchao.sparsity import apply_fake_sparsity, semi_sparse_weight, sparsify_
from torchao.utils import is_sm_at_least_90

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TestSemiStructuredSparse(common_utils.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @skipIfRocmVersionLessThan((7, 0))
    def test_sparse(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
            .eval()
        )

        apply_fake_sparsity(model)
        dense_result = model(input)

        sparsify_(model, semi_sparse_weight())
        sparse_result = model(input)

        if compile:
            model = torch.compile(model)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)


class TestBlockSparseWeight(common_utils.TestCase):
    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("input_shape", [1, 1024])
    def test_sparse(self, compile, input_shape):
        input = torch.rand((input_shape, 1024)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(1024, 2048),
                nn.Linear(2048, 1024),
            )
            .half()
            .cuda()
            .eval()
        )

        from torchao.sparsity.utils import create_block_sparse_tensor

        M, N = model[0].weight.shape
        model[0].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.float16)
        M, N = model[1].weight.shape
        model[1].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.float16)
        dense_result = model(input)

        from torchao.sparsity import block_sparse_weight

        sparsify_(model, block_sparse_weight(blocksize=64))
        # if compile:
        #     model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)


common_utils.instantiate_parametrized_tests(TestSemiStructuredSparse)
common_utils.instantiate_parametrized_tests(TestBlockSparseWeight)

if __name__ == "__main__":
    unittest.main()
