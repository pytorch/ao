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
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)
from torchao.sparsity import apply_fake_sparsity, semi_sparse_weight, sparsify_
from torchao.utils import is_sm_at_least_90

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TestSemiStructuredSparse(common_utils.TestCase):
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
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [False])
    @unittest.skip("Temporarily skip to unbreak CI")
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
        quantize_(model_copy, Int8DynamicActivationInt8WeightConfig())
        dense_result = model_copy(input)

        quantize_(
            model,
            Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()),
        )
        if compile:
            model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-2, atol=1e-2)

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
        quantize_(model_copy.bfloat16(), Int4WeightOnlyConfig(version=1))
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, Int4WeightOnlyConfig(layout=MarlinSparseLayout(), version=1))
        if compile:
            model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, atol=3e-1, rtol=3e-1)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [True, False])
    def test_fp8_cutlass_sparse(self, compile):
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
        quantize_(model_copy.bfloat16(), Float8DynamicActivationFloat8WeightConfig())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, Float8DynamicActivationFloat8SemiSparseWeightConfig())
        if compile:
            model = torch.compile(model)
        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, atol=3e-1, rtol=3e-1)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_fp8_cutlass_sparse_lowering_op_clone(self):
        with torch.inference_mode():
            model = nn.Linear(256, 1024).half().cuda().eval()
            apply_fake_sparsity(model)
            quantize_(model, Float8DynamicActivationFloat8SemiSparseWeightConfig())

            original = model.weight.original_weight_tensor.tensor_impl.get_plain()
            cloned = model.weight.original_weight_tensor.tensor_impl.clone().get_plain()

            for o, c in zip(original, cloned):
                torch.testing.assert_close(o, c, atol=0.0, rtol=0.0)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_fp8_cutlass_sparse_lowering_op_to(self):
        # Need to run with inference mode to avoid dispatching to `aten.to_copy`
        with torch.inference_mode():
            model = nn.Linear(256, 1024).half().cuda().eval()
            apply_fake_sparsity(model)
            model_copy = copy.deepcopy(model)
            expected = model_copy.weight.to(dtype=torch.float)

            quantize_(model, Float8DynamicActivationFloat8SemiSparseWeightConfig())

            original = torch.ops.aten.to.dtype_layout(
                model.weight.original_weight_tensor.tensor_impl,
                dtype=torch.float,
                layout=torch.strided,
            )
            torch.testing.assert_close(expected, original, atol=1e-1, rtol=1e-1)


class TestBlockSparseWeight(common_utils.TestCase):
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

        quantize_(model_copy, Int8DynamicActivationInt8WeightConfig())
        reference = model_copy(input)

        from torchao.dtypes import BlockSparseLayout

        quantize_(
            model,
            Int8DynamicActivationInt8WeightConfig(
                layout=BlockSparseLayout(blocksize=64)
            ),
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
