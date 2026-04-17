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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TestSemiStructuredSparse(common_utils.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @skipIfRocmVersionLessThan((7, 0))
    def test_sparse(self):
        if not torch.backends.cusparselt.is_available():
            self.skipTest("Need cuSPARSELt or hipSPARSELt")
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

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)


common_utils.instantiate_parametrized_tests(TestSemiStructuredSparse)

if __name__ == "__main__":
    unittest.main()
