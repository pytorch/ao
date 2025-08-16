# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from parameterized import param, parameterized

from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
    choose_qparams_and_quantize_codebook_coreml as choose_qparams_and_quantize_codebook_coreml_original,
)
from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
    choose_qparams_and_quantize_codebook_coreml_refactored,
    dequantize_codebook,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
)


class TestCoreMLQuantCompatibility(unittest.TestCase):
    TEST_CASES = [
        param(grouping_type="column", group_size=128, tensor_shape=(16, 1024)),
    ]

    @parameterized.expand(TEST_CASES)
    def test_functional_equivalence(self, grouping_type, group_size, tensor_shape):
        input_tensor = torch.randn(tensor_shape, dtype=torch.float32)
        code_dtype = torch.uint4
        nbits = _DTYPE_TO_BIT_WIDTH[code_dtype]
        torch.manual_seed(42)

        # --- Get results from reference implementations ---
        block_size = [-1, group_size]
        expected_luts, expected_codes = (
            choose_qparams_and_quantize_codebook_coreml_original(
                input_tensor, code_dtype, block_size.copy()
            )
        )

        actual_luts, actual_codes = (
            choose_qparams_and_quantize_codebook_coreml_refactored(
                input_tensor, code_dtype, block_size.copy()
            )
        )

        # Ensure codes are long for dequantize op compatibility
        expected_codes = expected_codes.to(torch.long)
        actual_codes = actual_codes.to(torch.long)

        self.assertEqual(
            actual_luts.shape,
            expected_luts.shape,
            "LUT shapes do not match after processing",
        )
        self.assertEqual(
            actual_codes.shape, expected_codes.shape, "Code shapes do not match"
        )

        dequant_expected = dequantize_codebook(
            expected_codes, expected_luts, nbits, block_size
        )
        dequant_actual = dequantize_codebook(
            actual_codes, actual_luts, nbits, block_size
        )

        expected_error = torch.mean((input_tensor - dequant_expected) ** 2).item()
        actual_error = torch.mean((input_tensor - dequant_actual) ** 2).item()

        self.assertAlmostEqual(
            actual_error,
            expected_error,
            delta=1e-5,
            msg="Dequantization error differs significantly between implementations",
        )


if __name__ == "__main__":
    unittest.main()
