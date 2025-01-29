# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch

from torchao.dtypes import PlainLayout
from torchao.experimental.q_dq_layout import QDQLayout
from torchao.experimental.quant_api import int8_dynamic_activation_intx_weight
from torchao.quantization.granularity import PerGroup, PerRow
from torchao.quantization.quant_api import quantize_
from torchao.utils import unwrap_tensor_subclass


class TestQDQLayout(unittest.TestCase):
    def test_accuracy(self):
        """
        Checks the accuracy of PackedLinearInt8DynamicActivationIntxWeightLayout() by comparing
        its results to the results of a reference model that uses PlainLayout()
        """
        granularity = PerGroup(128)
        m = 1
        n = 1071
        k = 4096
        activations = torch.randn(m, k)
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for weight_dtype in [
            torch.int1,
            torch.int2,
            torch.int3,
            torch.int4,
            torch.int5,
            torch.int6,
            torch.int7,
            torch.int8,
        ]:
            for has_weight_zeros in [True, False]:
                print(
                    f"Testing weight_dtype={weight_dtype}, has_weight_zeros={has_weight_zeros}"
                )
                quantized_model = copy.deepcopy(model)
                quantize_(
                    quantized_model,
                    int8_dynamic_activation_intx_weight(
                        weight_dtype=weight_dtype,
                        granularity=granularity,
                        has_weight_zeros=has_weight_zeros,
                        layout=QDQLayout(),
                    ),
                )

                quantized_model_reference = copy.deepcopy(model)
                quantize_(
                    quantized_model_reference,
                    int8_dynamic_activation_intx_weight(
                        weight_dtype=weight_dtype,
                        granularity=granularity,
                        has_weight_zeros=has_weight_zeros,
                        layout=PlainLayout(),
                    ),
                )

                with torch.no_grad():
                    result = quantized_model(activations)
                    expected_result = quantized_model_reference(activations)

                num_mismatch_at_low_tol = 0
                num_total = result.reshape(-1).shape[0]
                for i in range(num_total):
                    actual_val = result.reshape(-1)[i]
                    expected_val = expected_result.reshape(-1)[i]
                    self.assertTrue(torch.allclose(actual_val, expected_val, atol=1e-6))
                    if not torch.allclose(actual_val, expected_val):
                        num_mismatch_at_low_tol += 1

                # Assert at most 5% of entries are not close at a low tolerance
                self.assertTrue(num_mismatch_at_low_tol / num_total <= 0.05)

    def test_export(self):
        """
        Checks that models quantized with PackedLinearInt8DynamicActivationIntxWeightLayout() work with
        torch.export.export, torch.compile, and AOTI.
        """
        granularity = PerRow()
        m = 3
        k0 = 512
        k1 = 256
        k2 = 128
        k3 = 1024
        weight_dtype = torch.int4
        has_weight_zeros = True
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, k2, bias=False),
            torch.nn.Linear(k2, k3, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(2, 1, m, k0, dtype=torch.float32)

        print("Quantizing model")
        quantize_(
            model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=QDQLayout(),
            ),
        )
        eager_results = model(activations)

        unwrapped_model = copy.deepcopy(model)
        unwrap_tensor_subclass(model)

        print("Exporting quantized model")
        exported = torch.export.export(model, (activations,), strict=True)
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))


if __name__ == "__main__":
    unittest.main()
