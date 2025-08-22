# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch

from torchao.prototype.quantization.codebook_coreml import (
    CodebookQuantizedTensor,
    CodebookWeightOnlyConfig,
    choose_qparams_and_quantize_codebook_coreml,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error
from torchao.utils import is_package_at_least


@unittest.skipIf(
    not is_package_at_least("coremltools", "8.3.0"), "Requires coremltools >= 8.3.0"
)
class TestCodebookQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.input = torch.randn(100, 256, dtype=torch.float32)
        self.code_dtype = torch.uint8
        self.block_size = [-1, 4]
        self.nbits = 8

    def test_choose_qparams_codebook(self):
        codebook, wq = choose_qparams_and_quantize_codebook_coreml(
            self.input,
            self.code_dtype,
            self.block_size,
        )
        group_size = self.block_size[-1]
        self.assertEqual(codebook.shape, (1, 256 // group_size, 2**self.nbits, 1))
        self.assertEqual(wq.shape, (100, 256))

        self.assertFalse(torch.isnan(codebook).any())
        self.assertFalse(torch.isnan(wq).any())

    def test_codebook_quantized_tensor_from_float(self):
        cqt = CodebookQuantizedTensor.from_float(
            self.input,
            self.code_dtype,
            self.block_size,
        )

        dequant = cqt.dequantize()
        sqnr = compute_error(dequant, self.input)
        self.assertGreater(sqnr, 30)

    def test_codebook_quantized_tensor_from_float2(self):
        block_size = [-1, 16]
        code_dtype = torch.uint4

        cqt = CodebookQuantizedTensor.from_float(
            self.input,
            code_dtype,
            block_size,
        )

        dequant = cqt.dequantize()

        sqnr = compute_error(dequant, self.input)
        self.assertGreater(sqnr, 18)

    def test_quantize_api(self):
        m = torch.nn.Sequential(torch.nn.Linear(64, 64))
        quantize_(
            m,
            CodebookWeightOnlyConfig(dtype=self.code_dtype, block_size=self.block_size),
        )
        assert type(m[0].weight) == CodebookQuantizedTensor

    def test_choose_qparams_codebook_row_grouping(self):
        # Test with a block_size that forces row-wise grouping: [10, 256]
        # Input tensor is (100, 256)
        row_grouped_block_size = [10, -1]
        num_row_groups = (
            self.input.shape[0] // row_grouped_block_size[0]
        )  # 100 // 10 = 10

        codebook, wq = choose_qparams_and_quantize_codebook_coreml(
            self.input,
            self.code_dtype,
            row_grouped_block_size,
        )

        # Expected shape for row-wise grouping is (num_row_groups, 1, 2**nbits, 1)
        self.assertEqual(codebook.shape, (num_row_groups, 1, 2**self.nbits, 1))
        self.assertEqual(wq.shape, (100, 256))

        self.assertFalse(torch.isnan(codebook).any())
        self.assertFalse(torch.isnan(wq).any())

    def test_codebook_quantized_tensor_from_float_row_grouping(self):
        # Test end-to-end quantization/dequantization with row grouping
        row_grouped_block_size = [20, -1]  # 100 is divisible by 20
        cqt = CodebookQuantizedTensor.from_float(
            self.input,
            self.code_dtype,
            row_grouped_block_size,
        )

        dequant = cqt.dequantize()
        # The SQNR will be different from column grouping, but should still be high
        sqnr = compute_error(dequant, self.input)
        self.assertGreater(sqnr, 30)

    def test_export(self):
        m = torch.nn.Sequential(torch.nn.Linear(128, 64)).to(torch.float32)
        quantize_(m, CodebookWeightOnlyConfig(self.code_dtype, self.block_size))
        example_inputs = (torch.randn(1, 128, dtype=torch.float32),)
        m = torch.export.export(m, example_inputs).module()
        targets = [n.target for n in m.graph.nodes]
        self.assertTrue(torch.ops.quant.dequantize_codebook.default in targets)


if __name__ == "__main__":
    unittest.main()
