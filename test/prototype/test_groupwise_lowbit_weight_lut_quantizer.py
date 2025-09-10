# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
import torch.nn as nn
from parameterized import param, parameterized
from torch import uint1, uint2, uint3, uint4

from torchao.prototype.quantization.codebook_groupwise.api import (
    GroupwiseLutWeightConfig,
)
from torchao.prototype.quantization.codebook_utils.codebook_utils import (
    group_size_to_block_shapes,
)
from torchao.quantization.quant_api import quantize_
from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    _is_kernel_library_loaded,
)


@unittest.skipIf(not _is_kernel_library_loaded(), "Need torchao lowbit kernels")
class TestGroupwiseLowbitWeightLut(unittest.TestCase):
    """
    Test suite for the GroupwiseLutWeight quantization scheme, updated for the
    new simplified API.
    """

    TEST_CASES = [
        param(
            code_dtype=code_dtype,
            lut_group_size=lut_group_size,
            weight_dtype=weight_dtype,
            has_bias=has_bias,
        )
        for code_dtype in [uint1, uint2, uint3, uint4]
        for lut_group_size in [256, 512]
        for weight_dtype in [torch.float32]
        for has_bias in [True, False]
    ]

    # --------------------------------------------------------------------------
    # Test 1: End-to-End Model Accuracy
    # --------------------------------------------------------------------------
    @parameterized.expand(TEST_CASES)
    def test_e2e_accuracy_vs_reference(
        self,
        code_dtype,
        lut_group_size,
        weight_dtype,
        has_bias,
    ):
        """
        Tests the numerical accuracy of the full quantized model against a reference.
        This now uses the `use_qdq_reference` flag instead of layout objects.
        """
        m, k, n = 3, 64, 32
        activations = torch.randn(m, k, dtype=weight_dtype)
        model = nn.Sequential(nn.Linear(k, n, bias=has_bias)).to(dtype=weight_dtype)

        # --- 2. Update tensor_shape to reflect the new (k, n) layout ---
        lut_block_shape = group_size_to_block_shapes(
            lut_group_size=lut_group_size, tensor_shape=(n, k)
        )

        # --- Quantize using C++ ops ---
        quantized_model = copy.deepcopy(model)
        perf_config = GroupwiseLutWeightConfig(
            code_dtype=code_dtype,
            weight_dtype=weight_dtype,
            lut_block_shape=lut_block_shape,
            use_qdq_reference=False,
        )
        quantize_(quantized_model, perf_config)
        with torch.no_grad():
            actual_result = quantized_model(activations)

        # --- Quantize for Reference (using Python ops) ---
        reference_model = copy.deepcopy(model)
        ref_config = GroupwiseLutWeightConfig(
            code_dtype=code_dtype,
            weight_dtype=weight_dtype,
            lut_block_shape=lut_block_shape,
            use_qdq_reference=True,
        )
        quantize_(reference_model, ref_config)
        with torch.no_grad():
            expected_result = reference_model(activations)
        # Compare results
        self.assertTrue(
            torch.allclose(actual_result, expected_result, atol=1e-2, rtol=1e-2)
        )

    def tearDown(self):
        """
        Clear the TorchDynamo cache after each test case to prevent
        recompilation errors in parameterized tests.
        """
        super().tearDown()
        torch._dynamo.reset()

    # --------------------------------------------------------------------------
    # Test 2: Deployment Readiness (Updated for new API)
    # --------------------------------------------------------------------------
    @parameterized.expand(TEST_CASES)
    def test_export_compile_aoti(
        self,
        code_dtype,
        lut_group_size,
        weight_dtype,
        has_bias,
    ):
        """
        Tests that the quantized model can be exported and compiled.
        """
        k, n = 64, 32
        activations = torch.randn(2, k, dtype=weight_dtype)
        model = (
            nn.Sequential(nn.Linear(k, n, bias=has_bias)).to(dtype=weight_dtype).eval()
        )
        lut_block_shape = group_size_to_block_shapes(
            lut_group_size=lut_group_size,
            tensor_shape=(n, k),
        )

        # Configure the quantization using the new API
        config = GroupwiseLutWeightConfig(
            code_dtype=code_dtype,
            weight_dtype=weight_dtype,
            lut_block_shape=lut_block_shape,
            use_qdq_reference=False,
        )
        quantize_(model, config)

        with torch.no_grad():
            eager_results = model(activations)

        # Export and Compile
        exported_model = torch.export.export(model, (activations,))
        compiled_model = torch.compile(model, fullgraph=True)

        with tempfile.TemporaryDirectory() as tmpdir, torch.no_grad():
            # Check exported model
            exported_results = exported_model.module()(activations)
            self.assertTrue(
                torch.allclose(eager_results, exported_results, atol=1e-3, rtol=1e-3)
            )

            # Check compiled model
            compiled_results = compiled_model(activations)
            self.assertTrue(
                torch.allclose(eager_results, compiled_results, atol=1e-3, rtol=1e-3)
            )

            # Check AOTI compiled model using the packaging API
            package_path = f"{tmpdir}/model.pt2"
            torch._inductor.aoti_compile_and_package(
                exported_model, package_path=package_path
            )
            aoti_model = torch._inductor.aoti_load_package(package_path)
            aoti_results = aoti_model(activations)
            self.assertTrue(
                torch.allclose(eager_results, aoti_results, atol=1e-3, rtol=1e-3)
            )


if __name__ == "__main__":
    unittest.main()
