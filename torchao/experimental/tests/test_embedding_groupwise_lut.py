# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
import torch.nn as nn
from parameterized import param, parameterized
from torch import uint1, uint2, uint3, uint4

from torchao.prototype.quantization.codebook_groupwise.api import (
    EmbeddingLutQuantizer,
    GroupwiseLutWeightConfig,
)


def generate_test_cases():
    """Generates test cases with logic to handle has_scales correctly."""
    code_dtypes = [uint1, uint2, uint3, uint4]
    lut_block_shapes = [[1, -1], [2, -1], [4, -1]]

    test_cases = []

    for code_dtype in code_dtypes:
        for lut_block_shape in lut_block_shapes:
            test_cases.append(
                param(
                    config=GroupwiseLutWeightConfig(
                        code_dtype=code_dtype,
                        lut_block_shape=lut_block_shape,
                        scale_block_shape=None,
                        has_scale=False,
                    ),
                    embedding_dim=256,
                    num_embeddings=128,
                )
            )

    return test_cases


class TestLutEmbeddingQuantizer(unittest.TestCase):
    @parameterized.expand(generate_test_cases())
    def test_accuracy_vs_qdq_reference(
        self,
        config: GroupwiseLutWeightConfig,
        embedding_dim: int,
        num_embeddings: int = 128,
    ):
        """
        Tests the numerical accuracy of the custom quantized embedding module
        against a QDQ (Quantize-Dequantize) reference implementation.
        """
        embedding_dim = embedding_dim
        model = nn.Sequential(nn.Embedding(num_embeddings, embedding_dim))
        indices = torch.randint(0, num_embeddings, (10, 20), dtype=torch.int64)

        # --- 1. Get ACTUAL result from the custom kernel implementation ---
        quantized_model = copy.deepcopy(model)
        # Ensure the 'use_qdq_reference' flag is False for the performance path
        perf_config = copy.deepcopy(config)
        perf_config.use_qdq_reference = False

        quantizer = EmbeddingLutQuantizer(perf_config)
        quantizer.quantize(quantized_model)

        with torch.no_grad():
            actual_result = quantized_model(indices)

        # --- 2. Get EXPECTED result from the QDQ reference implementation ---
        reference_model = copy.deepcopy(model)
        # Set the 'use_qdq_reference' flag to True for the reference path
        ref_config = copy.deepcopy(config)
        ref_config.use_qdq_reference = True

        quantizer = EmbeddingLutQuantizer(ref_config)
        quantizer.quantize(reference_model)

        with torch.no_grad():
            expected_result = reference_model(indices)

        # --- 3. Compare results ---
        self.assertTrue(
            torch.allclose(actual_result, expected_result, atol=1e-6, rtol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
