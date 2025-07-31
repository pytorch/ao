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
from torchao.testing.utils import skip_if_no_cuda
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6, is_package_at_least


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
            [self.input.shape[0], 4],
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

    @skip_if_no_cuda()
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "requires 2.6+.")
    def test_export(self):
        m = torch.nn.Sequential(torch.nn.Linear(128, 64)).to(torch.float32)
        quantize_(m, CodebookWeightOnlyConfig(self.code_dtype, self.block_size))
        example_inputs = (torch.randn(1, 128, dtype=torch.float32),)
        m = torch.export.export(m, example_inputs).module()
        targets = [n.target for n in m.graph.nodes]
        self.assertTrue(torch.ops.quant.dequantize_codebook.default in targets)


if __name__ == "__main__":
    unittest.main()
