# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import unittest

import pytest
import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.utils import auto_detect_device

_DEVICE = auto_detect_device()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TestSupermask(common_utils.TestCase):
    @common_utils.parametrize("sparsity_level", [0.25, 0.5])
    @common_utils.parametrize("blocksize", [2, 4, 8])
    def test_supermask(self, sparsity_level, blocksize):
        model = (
            nn.Sequential(
                nn.Linear(16, 16, bias=False),
            )
            .half()
            .to(_DEVICE)
            .eval()
        )

        from torchao.sparsity import SupermaskLinear

        M, N = model[0].weight.shape
        model[0] = SupermaskLinear.from_linear(
            model[0], sparsity_level=sparsity_level, blocksize=blocksize
        )
        model[0] = SupermaskLinear.to_linear(model[0])
        weight_bsr = model[0].weight.to_sparse_bsr(blocksize=blocksize)

        # Test correct sparsity level
        nnz = weight_bsr._nnz()
        expected = round((M // blocksize) * (N // blocksize) * (1 - sparsity_level))
        assert nnz == expected, f"Expected {expected} nonzeros, got {nnz}"

    def test_from_linear(self):
        from torchao.sparsity import SupermaskLinear

        linear = nn.Linear(128, 128)
        supermask_linear = SupermaskLinear.from_linear(
            linear, sparsity_level=0.5, blocksize=4
        )
        assert supermask_linear.weight.shape == linear.weight.shape


common_utils.instantiate_parametrized_tests(TestSupermask)

if __name__ == "__main__":
    unittest.main()
