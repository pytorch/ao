import copy
import logging
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.dtypes import MarlinSparseLayout, SemiSparseLayout
from torchao.quantization.quant_api import (
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
    quantize_,
)
from torchao.sparsity import apply_fake_sparsity, semi_sparse_weight, sparsify_
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TestBlockSparseWeight(common_utils.TestCase):
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4,
        "pytorch 2.4+ feature due to need for custom op support",
    )
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [True, False])
    def test_sparse(self, compile):
        input = torch.rand((1024, 1024)).half().cuda()
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

        from torchao.prototype.sparsity.superblock.blocksparse import (
            block_sparse_weight,
        )

        sparsify_(model, block_sparse_weight(blocksize=64))
        # if compile:
        #     model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

common_utils.instantiate_parametrized_tests(TestBlockSparseWeight)

if __name__ == "__main__":
    unittest.main()
