# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
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


class TestSemiStructuredSparse(common_utils.TestCase):
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_3, "pytorch 2.3+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skip("Temporarily skipping to unpin nightlies")
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


class TestQuantSemiSparse(common_utils.TestCase):
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_5, "pytorch 2.5+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [False])
    def test_quant_semi_sparse(self, compile):
        if not torch.backends.cusparselt.is_available():
            self.skipTest("Need cuSPARSELt")

        # compile True failed with CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit(...
        # https://github.com/pytorch/ao/actions/runs/11978863581/job/33402892517?pr=1330

        torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False

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
        model_copy = copy.deepcopy(model)
        quantize_(model_copy, int8_dynamic_activation_int8_weight())
        dense_result = model_copy(input)

        quantize_(
            model,
            int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()),
        )
        if compile:
            model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_5, "pytorch 2.5+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [True, False])
    def test_sparse_marlin(self, compile):
        if not torch.backends.cusparselt.is_available():
            self.skipTest("Need cuSPARSELt")

        input = torch.rand((256, 256)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(256, 1024),
                nn.Linear(1024, 256),
            )
            .half()
            .cuda()
            .eval()
        )

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout=MarlinSparseLayout()))
        if compile:
            model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, atol=3e-1, rtol=3e-1)


class TestBlockSparseWeight(common_utils.TestCase):
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_4,
        "pytorch 2.4+ feature due to need for custom op support",
    )
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


class TestQuantBlockSparseWeight(common_utils.TestCase):
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "pytorch 2.6+ feature")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [True, False])
    def test_sparse(self, compile):
        input = torch.rand((256, 128)).to(torch.bfloat16).cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .to(torch.bfloat16)
            .cuda()
            .eval()
        )
        from torchao.sparsity.utils import create_block_sparse_tensor

        M, N = model[0].weight.shape
        model[0].weight.data = (
            create_block_sparse_tensor(M, N, 64, 0.5, torch.bfloat16)
            * torch.rand(M, N, dtype=torch.bfloat16).cuda()
        )
        M, N = model[1].weight.shape
        model[1].weight.data = create_block_sparse_tensor(M, N, 64, 0.5, torch.bfloat16)

        model_copy = copy.deepcopy(model)

        quantize_(model_copy, int8_dynamic_activation_int8_weight())
        reference = model_copy(input)

        from torchao.dtypes import BlockSparseLayout

        quantize_(
            model,
            int8_dynamic_activation_int8_weight(layout=BlockSparseLayout(blocksize=64)),
        )
        if compile:
            model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(reference, sparse_result, rtol=1e-1, atol=1e-1)


common_utils.instantiate_parametrized_tests(TestSemiStructuredSparse)
common_utils.instantiate_parametrized_tests(TestQuantSemiSparse)
common_utils.instantiate_parametrized_tests(TestBlockSparseWeight)
common_utils.instantiate_parametrized_tests(TestQuantBlockSparseWeight)

if __name__ == "__main__":
    unittest.main()
